import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # Progress bar (optional)

learning_rate = 2e-5
batch_size = 8
num_train_epochs = 3

model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#***Prepare the dataset***
import torch
from transformers import PreTrainedTokenizer
import pyarrow.parquet as pq

class CodeGenDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer: PreTrainedTokenizer):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer

    def _load_data(self, data_path):
        # Load data from Parquet using pyarrow
        data = pq.read_table(data_path)

        return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print('Data Type:', self.data.shape)
        #print(idx)
        prompt = self.data.take([idx]).column('instruction').to_numpy()[0]
        code = self.data.take([idx]).column('output').to_numpy()[0]
        #print('Prompt:', prompt, 'Type:', type(prompt))
        #print('Code:', code, 'Type:', type(code))

        # Preprocess input and output using the tokenizer
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        labels = self.tokenizer.encode(code, add_special_tokens=True, return_tensors="pt")

        # Adjust for padding and masking (optional)
        #input_ids = self._pad(input_ids)
        #labels = self._pad(labels)

        print('input_ids:', input_ids.shape)
        print('attention_mask:', input_ids.ne(self.tokenizer.pad_token_id).shape)

        padded_data = self.tokenizer(input_ids=input_ids, return_tensors="pt", padding=True, add_special_tokens=True)
        
        #input_ids = padded_data['input_ids']
        #attention_mask = padded_data['attention_mask']



        return {
            'input_ids': padded_data['input_ids'],
            'attention_mask': padded_data['attention_mask'],  # Attention mask for padding
            'labels': labels
        }
        


    def _pad(self, sequence, padding_value=0):
        # This example pads sequences to the maximum length in the batch
        max_len = max(len(seq) for seq in self.data)
        print("Length before padding:", sequence.shape)
        padded_sequence = torch.nn.functional.pad(sequence, (0, max_len - len(sequence)), value=padding_value)
        print("max_len:", max_len, "padded sequence len:", padded_sequence.shape)
        print("padded_sequence", padded_sequence, '\n')
        return torch.nn.functional.pad(sequence, (0, max_len - sequence.shape[1]), value=padding_value)
    
code_120k = CodeGenDataset("../train-00000-of-00001-d9b93805488c263e.parquet", tokenizer)

print(len(code_120k))


# Replace "your_dataset" with your actual dataset object
train_dataloader = DataLoader(code_120k, batch_size=batch_size, shuffle=True)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Adjust for masking

model.train()  # Set model to training mode

for epoch in range(num_train_epochs):
    for data in tqdm(train_dataloader):
        input_ids = data['input_ids'].to(model.device)
        attention_mask = data['attention_mask'].to(model.device)
        print('Attention Mask:', attention_mask.shape)
        labels = data['labels'].to(model.device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state

        # Loss calculation
        loss = criterion(logits.view(-1, tokenizer.vocab_size), labels.view(-1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # (Optional) Track and log training metrics (loss, accuracy)

# After training, save the fine-tuned model for future use
model.save_pretrained("my_fine-tuned_deepseek_coder")
tokenizer.save_pretrained("my_fine-tuned_deepseek_coder")

