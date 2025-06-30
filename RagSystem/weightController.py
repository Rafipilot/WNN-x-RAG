### This script dynamically adjust the weights of items in the vector database based on their relevance to the input embedding.

import ao_core as ao
from config import openai_key
import ao_embeddings.binaryEmbeddings as be
from RagSystem.Vectorizer import vectorizer
import json

class weightController:
    def __init__(self):
        self.Arch = ao.Arch(arch_i=[10,4,4], arch_z=[1,1,1,1])

        self.Agent = ao.Agent(Arch=self.Arch)

        self.em = be.binaryEmbeddings(openai_api_key=openai_key, numberBinaryDigits=10)

    def convert_to_binary(self, interger):
        binary = format(int(interger), '04b')
        return [int(bit) for bit in binary]

    def convert_to_int(self, binary):
        binary_str = ''.join(str(bit) for bit in binary)
        return int(binary_str, 2)

    def adjust_weights(self, vector_db, vectorizer: vectorizer):
        for entry in vector_db:
            
            binary_embedding = self.em.embeddingToBinary(entry["embedding"])

            number_of_retrievals = entry["numberOfRetrievals"]
            number_of_retrievals_binary = self.convert_to_binary(number_of_retrievals)

            weight = int(entry["weight"]*10)
            weight = self.convert_to_binary(weight)

            print("weight:", weight)
            print("binary embedding:", binary_embedding)
            print("number of retrievals binary:", number_of_retrievals_binary)

            input_to_agent = binary_embedding + number_of_retrievals_binary + weight

            self.most_recent_input = input_to_agent

            print("most_recent_input:", self.most_recent_input)

            print("input to agent:", input_to_agent)

            new_weight = self.convert_to_int(self.Agent.next_state(input_to_agent))/10

            entry["weight"] = new_weight
            print(f"Updated weight for '{entry['input']}': {new_weight}")

            # Save the updated vector database
            vectorizer.save_cache()
            #
    def train_agent(self, type):
        if type == "pos":
            self.Agent.next_state(INPUT=self.most_recent_input, LABEL=[1,1,1,1])
        else:
            self.Agent.next_state(INPUT=self.most_recent_input, LABEL=[0,0,0,0])

            
