import pandas as pd
import torch
import huggingface_hub
from datasets import Dataset
import transformers
from transformers import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import gc
import os

class CharacterChatBot:
    def __init__(self, model_path, data_path="data/blackclover.csv", huggingface_token=None):
        self.model_path = model_path
        self.data_path = data_path
        self.huggingface_token = huggingface_token
        self.base_model_path = "meta-llama/Llama-3.1-8B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.huggingface_token:
            huggingface_hub.login(self.huggingface_token)
        
        if self.check_model_availability(self.model_path):
            self.model = self.load_model(self.model_path)
        else:
            print("Model not found in Hugging Face Hub; training a new model.")
            train_dataset = self.load_data()
            self.train(self.base_model_path, train_dataset)
            self.model = self.load_model(self.model_path)

    def check_model_availability(self, model_path):
        try:
            model_info = huggingface_hub.model_info(model_path, token=self.huggingface_token)
            return "model_type" in model_info.config
        except Exception as e:
            print(f"Error accessing model {model_path}: {e}")
            return False

    def chat(self, message, history):
        messages = [{"role": "system", "content": """You are a character from the anime "Black Clover". Your responses should reflect the personality and speech patterns of the character.\n"""}]
        
        for user_message, bot_response in history:
            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": bot_response})
        
        messages.append({"role": "user", "content": message})
        
        output = self.model(
            messages,
            max_length=256,
            eos_token_id=self.model.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        output_message = output[0]['generated_text'][-1]
        return output_message

    def load_model(self, model_path):
        try:
            if torch.cuda.is_available():
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_path,
                    model_kwargs={"torch_dtype": torch.float16, "quantization_config": bnb_config}
                )
            else:
                pipeline = transformers.pipeline("text-generation", model=model_path)
            return pipeline
        except Exception as e:
            raise ValueError(f"Error loading model '{model_path}': {e}")

    def load_data(self):
        column_names = ['speaker', 'line']
        
        try:
            blackclover_transcript_df = pd.read_csv(self.data_path, names=column_names, header=None)
            print("First few rows of the loaded DataFrame:")
            print(blackclover_transcript_df.head())
            
            blackclover_transcript_df.dropna(subset=['speaker', 'line'], inplace=True)
            blackclover_transcript_df['number_of_words'] = blackclover_transcript_df['line'].apply(lambda x: len(str(x).split()))

            characters_of_interest = ['Orsi', 'Men', 'Asta', 'Lily', 'Recca', 'Nash', 'Yuno', 'Aruru', 'Narrator', 'Noble 1', 'Noble 2', 'Drouot', 'Crowd', 'Children', 'Revchi']
            blackclover_transcript_df['blackclover_response_flag'] = 0
            blackclover_transcript_df.loc[
                (blackclover_transcript_df['speaker'].isin(characters_of_interest)) & (blackclover_transcript_df['number_of_words'] > 5),
                'blackclover_response_flag'
            ] = 1

            indexes_to_take = blackclover_transcript_df[blackclover_transcript_df['blackclover_response_flag'] == 1].index.tolist()
            system_prompt = """You are a character from the anime "Black Clover". Your responses should reflect the personality and speech patterns of the character.\n"""
            prompts = [
                system_prompt + blackclover_transcript_df.iloc[i - 1]['line'] + "\n" + blackclover_transcript_df.iloc[i]['line']
                for i in indexes_to_take if i > 0
            ]

            df = pd.DataFrame({"prompt": prompts})
            return Dataset.from_pandas(df)
        
        except Exception as e:
            print("An error occurred while loading data:", e)
            raise

    def train(self, base_model_name_or_path, dataset, output_dir="./results"):
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, quantization_config=bnb_config, trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, trust_remote_code=True)

        optimizer = AdamW(model.parameters(), lr=2e-4)  # use AdamW as an alternative

        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

        peft_config = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CASUAL_LM")
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="prompt",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=SFTConfig(output_dir=output_dir, per_device_train_batch_size=1, gradient_accumulation_steps=1, optim="paged_adamw_32bit", save_steps=200, logging_steps=10, learning_rate=2e-4, fp16=True, max_grad_norm=0.3, max_steps=300, warmup_ratio=0.3, group_by_length=True, lr_scheduler_type="constant", report_to="none")
        )
        
        trainer.train()
        trainer.model.save_pretrained("final_ckpt")
        tokenizer.save_pretrained("final_ckpt")
        del trainer, model
        gc.collect()

        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=torch.float16, device_map=self.device)
        model = PeftModel.from_pretrained(base_model, "final_ckpt")
        model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)
        del model, base_model
        gc.collect()
