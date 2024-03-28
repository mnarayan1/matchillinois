import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
import gensim
from bertopic import BERTopic
import warnings
import csv

# NLP processing setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    return " ".join(word for word in words if word.lower() not in stop_words)

def process_teams(input_csv_path, output_csv_path):
    data = pd.read_csv(input_csv_path)
    list_csv_files = []
    likert_columns = ["Web Dev", "ML", "Mobile Dev", "Game Dev", "Data Analytics"]
    weights = {1: 0.5, 2: 0.4, 3: 0.05, 4: 0.03, 5: 0.02}
    probabilities = data[likert_columns].apply(lambda x: [weights[val] for val in x], axis=1)
    group_assignments = {}

    for idx, row in data.iterrows():
        topic_probabilities = [weights[val] for val in row[likert_columns]]
        user_id = row['User']
        assigned_topic_index = np.random.choice(len(topic_probabilities), p=topic_probabilities)
        assigned_topic = likert_columns[assigned_topic_index]
        group_assignments[user_id] = assigned_topic

    for topic in likert_columns:
        topic_data = [(user_id, data.loc[data['User'] == user_id, 'Project Description'].values[0])
                      for user_id, assigned_topic in group_assignments.items() if assigned_topic == topic]
        group_csv = f"{topic}_group.csv"
        pd.DataFrame(topic_data, columns=['User', 'Project Description']).to_csv(group_csv, index=False)
        list_csv_files.append(group_csv)


    def nlp_processing(file_name):
        data = pd.read_csv(file_name)
        data['Project Description'] = data['Project Description'].apply(remove_stopwords)
        texts = data['Project Description'].tolist()

        # Initialize LDATopic column to a default value
        data['LDATopic'] = -1

        # Fit BERTopic model
        bertopic_model = BERTopic(verbose=True, embedding_model='paraphrase-MiniLM-L3-v2', min_topic_size=5)
        bertopic_topics, _ = bertopic_model.fit_transform(texts)
        data['BERTopic'] = bertopic_topics

        # Filter out outliers marked with -1 by BERTopic
        outlier_data = data[data['BERTopic'] == -1]
        outlier_texts = outlier_data['Project Description'].tolist()
        if not outlier_texts:
            # Sort the data by the topic
            sorted_data = data.sort_values(by='BERTopic')
            sorted_data.to_csv(f"NLP_{file_name}", index=False)

        else: 
            # Preprocess and vectorize texts for LDA
            vectorizer = CountVectorizer(stop_words='english')
            X = vectorizer.fit_transform(outlier_texts)
            corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
            dictionary = corpora.Dictionary.from_corpus(corpus, id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))

            # Fit LDA model on the outliers
            lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)
            outlier_data['LDATopic'] = [max(lda_model[corpus[i]], key=lambda x: x[1])[0] for i in range(len(outlier_texts))]

            # Combine the LDA topic assignments for outliers with the original dataset
            data.loc[data['BERTopic'] == -1, 'LDATopic'] = outlier_data['LDATopic']

            # Sort the data first by BERTopic (excluding -1), then by LDATopic for outliers
            non_outlier_data = data[data['BERTopic'] != -1].sort_values(by='BERTopic')
            outlier_data_sorted = outlier_data.sort_values(by='LDATopic')
            sorted_data = pd.concat([non_outlier_data, outlier_data_sorted])

            # Save the sorted data to a CSV file
            sorted_data.to_csv(f"NLP_{file_name}", index=False)
        return f"NLP_{file_name}"

    nlp_csv_files = [nlp_processing(file) for file in list_csv_files]

    def create_teams(grouped_users):
        teams = []
        small_groups = []

        for group in grouped_users:
            while len(group) > 6:
                teams.append(group[:6])
                group = group[6:]
            if len(group) >= 3:
                teams.append(group)
            else:
                small_groups.extend(group)

        for member in small_groups:
            added_to_team = False
            for team in teams:
                if len(team) < 6:
                    team.append(member)
                    added_to_team = True
                    break
            if not added_to_team:
                teams.append([member])

        return teams

    final_data = []
    with open(output_csv_path, "w", newline='') as outfile:
        csv_writer = csv.writer(outfile)
        
        # Write a header row for the whole file (this makes it clear these are data columns, not part of the data)
        csv_writer.writerow(["Group", "Team Number", "User ID", "Project Description"])  # You can adjust these headers as needed

        for file in nlp_csv_files:
            # Extract the group name from the filename
            group_name = file.split('_')[1]  # This gets the second word in the filename

            # Process each file for team formation
            data = pd.read_csv(file)
            grouped_lists = data.groupby('BERTopic').apply(lambda x: x['User'].tolist()).tolist()
            final_teams = create_teams(grouped_lists)

            # Iterate over the teams and write their information
            for i, team in enumerate(final_teams):
                for user in team:
                    user_data = data[data['User'] == user].iloc[0]
                    # Include the group name in each row of data
                    team_data = [group_name, i + 1, user, user_data['Project Description']]  # Define row data
                    csv_writer.writerow(team_data)  # Write the row data
                
                # Optionally, you can write an empty row after each team for better readability
                csv_writer.writerow([])  # Adds a space between teams for clarity

            # Write a couple of newline characters to separate each group's section
            csv_writer.writerow([])  # Adds a space between different groups