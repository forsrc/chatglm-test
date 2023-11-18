import torch
import sqlite3
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime





class Chat:
    def __init__(self, model_name, db_name, knowledge_file, checkpoint_dir):
        
        self.model_name = model_name
        self.db_name = os.path.abspath(db_name)
        self.tokenizer_state_dict = os.path.abspath(os.path.join(checkpoint_dir, "tokenizer_state_dict"))
        self.checkpoint_file = os.path.abspath(os.path.join(checkpoint_dir, "checkpoint.pt"))
      
        self.knowledge_file = os.path.abspath(knowledge_file)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        if self.device.type == 'cuda' or self.device.type == 'mps':
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).float() #cpu

        self.model =  self.model.eval()
        self.history = [("あなたは私の個人的な IT アシスタントです。これからはすべての質問に日本語で答えます。", "")]
        
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.create_table()
        
        if self.checkpoint_file:
            self.load_checkpoint()
        
        self.knowledge_base = ""
        if knowledge_file: 
            self.knowledge_base = self.load_knowledge_base()
            #self.history.append(self.knowledge_base)
        
        self.history.extend([(self.knowledge_base, "")])

    def input(self, message):
        print("--> " + message)

        prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                bos_token=self.tokenizer.bos_token,
                b_inst="<INST>",
                system="""
                <SYS>
                あなたはITのアシスタントです。すべての質問には日本語で回答します。
                </SYS>
                """,
                prompt=message,
                e_inst="</INST>",
        )
        response, history = self.model.chat(self.tokenizer, message, history=self.history)
        self.history = history
        #print("==> " + str(history))
        print("==> " + response)
        return response
        

    def message(self, message):
        print("--> " + message)
        
        prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
                bos_token=self.tokenizer.bos_token,
                b_inst="<INST>",
                system="""
                <SYS>
                あなたはITのアシスタントです。すべての質問には日本語で回答します。
                </SYS>
                """,
                prompt=message,
                e_inst="</INST>",
        )


        with torch.no_grad():
            input_ids = self.tokenizer(message, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        
            output_ids = self.model.generate(input_ids,
                                             max_length=5000,
                                             num_return_sequences=1,
                                             pad_token_id=self.tokenizer.pad_token_id,
                                             eos_token_id=self.tokenizer.eos_token_id,
                                             )
        
        response = self.tokenizer.decode(output_ids.tolist()[0][input_ids.size(1) :], skip_special_tokens=True)

        self.history.append(response)
        
        self.insert_qa(prompt, response)

        print("==> " + response)
        return response


    def load_knowledge_base(self):
        knowledge_base = ""

        with open(self.knowledge_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                knowledge_base += line

        print("--> " + self.knowledge_file + " -> " + str(knowledge_base))
        return knowledge_base

    def save_checkpoint(self):
        print("Saving checkpoint...")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "tokenizer_state_dict": self.tokenizer.save_pretrained(self.tokenizer_state_dict),
        }
        torch.save(checkpoint, self.checkpoint_file)
        print("Saving checkpoint... OK")

    def load_checkpoint(self):
        if os.path.isfile(self.checkpoint_file):
            print("Loading checkpoint...")
            checkpoint = torch.load(self.checkpoint_file)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if hasattr(self.tokenizer, 'load_state_dict'):
                self.tokenizer.load_state_dict(checkpoint["tokenizer_state_dict"])
            print("Loading checkpoint... OK")


    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT
            )
        ''')
        self.conn.commit()

    def insert_qa(self, question, answer):
        cursor = self.conn.cursor()
        cursor.execute('INSERT INTO qa (question, answer) VALUES (?, ?)', (question, answer))
        self.conn.commit()
        
    def close(self):
        print("close db ...")
        self.conn.close()
        print("close db ... OK")
        #self.save_checkpoint()
