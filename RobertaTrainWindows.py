import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from transformers import RobertaTokenizer
import pandas as pd
import textwrap
torch.manual_seed(0)


if os.path.exists('/home/sslater/data/trainwindowsimproved.csv'):
    data_path = '/home/sslater/data/trainwindowsimproved.csv'
else:
    data_path = '/Users/stevenslater/Desktop/FinalProject/Data/trainwindowsimproved.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df = pd.read_csv(data_path)

'''
# Separate the dataset into positive and negative classes
positive_df = train_df[train_df['labels'] == 1]
negative_df = train_df[train_df['labels'] == 0]

# Perform Random Oversampling on the positive class
oversampled_positive_df = positive_df.sample(n=len(negative_df), replace=True)

# Combine and shuffle the datasets
balanced_df = pd.concat([negative_df, oversampled_positive_df]).sample(frac=1).reset_index(drop=True)
#print balanced df value counts
print(balanced_df['labels'].value_counts())
'''

# Define the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# Add a classifier on top of the model -testing
classifier = nn.Linear(model.config.hidden_size, 1)
sigmoid = nn.Sigmoid()

#PUT THEM ON THE GPU
model = model.to(device)
classifier = classifier.to(device)

# Calculate the weight for the positive class
#there is 13 times more neg than pos, but just start with 5 for now
weight_for_pos = 2
#print(weight_for_pos)
pos_weight = torch.tensor([weight_for_pos], dtype=torch.float32).to(device)

# EITHER PASS WEIGHT FOR WEIGHING POS OR NOTHING
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#loss_function = nn.BCEWithLogitsLoss() #this is used for when no weights

#Learning Rate
optimizer = optim.Adam(model.parameters(), lr=0.00002) #lr =0.1


# Define the run function with DataFrame as a parameter
def run(train_df, batch_size=32, num_epochs=1):
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")

        # Optional: Shuffle the dataset
        df_shuffled = train_df.sample(frac=1).reset_index(drop=True)

        # Process the conversations in batches
        for i in tqdm(range(0, len(train_df), batch_size), desc="Processing Batches"):
            # Extract the batch of conversation chunks
            batch = df_shuffled.iloc[i:i + batch_size]

            # Initialize the gradients
            optimizer.zero_grad()

            # Initialize a variable to store the total loss for the batch
            total_loss = 0.0

            # Process each conversation chunk separately
            for _, row in batch.iterrows():
                # Extract the conversation chunk and its label
                conversation_chunk = row['conversation_chunks']
                label = row['labels']

                # Tokenize the conversation chunk and convert it into a tensor
                input_ids = tokenizer.encode(conversation_chunk, add_special_tokens=True, return_tensors='pt').to(device)
                
                # Pass the tensor to the model
                output = model(input_ids)[0]

                # Take the hidden state corresponding to the first token
                output = output[:, 0, :]

                # Use the classifier to make a prediction for the chunk
                logits = classifier(output)
                logits = logits.squeeze(-1)

                # Compute the loss between the prediction and the label
                label_tensor = torch.tensor([label], dtype=torch.float).to(device)
                loss = loss_function(logits, label_tensor)

                # Add the loss to the total loss
                total_loss += loss

            # Average the loss over the batch
            average_loss = total_loss / len(batch)
            # Backward pass for the whole batch
            average_loss.backward()
            optimizer.step()
            print(f"Average loss for this batch: {average_loss.item()}")



# Call the run function with the DataFrame
run(train_df)

# Save the model parameters
torch.save(model.state_dict(), 'newWindowsRoberta1epochWeight2.pth')



