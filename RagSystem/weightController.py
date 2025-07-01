### This script dynamically adjust the weights of items in the vector database based on their relevance to the input embedding.

import ao_core as ao
from config import openai_key
import ao_embeddings.binaryEmbeddings as be
from RagSystem.Vectorizer import vectorizer

class weightController:
    def __init__(self, vectorizer: vectorizer):

        self.vectorizer = vectorizer
        self.vector_db = vectorizer.cache
        self.Arch = ao.Arch(arch_i=[10,4,4], arch_z=[1,1,1,1]) # Input is condensed embedding, number of retrivals, current weight. Output is the next weight # TODO add a unique identifier

        self.Agent = ao.Agent(Arch=self.Arch)

        self.em = be.binaryEmbeddings(openai_api_key=openai_key, numberBinaryDigits=10)

    def convert_to_binary(self, interger):
        binary = format(int(interger), '04b')
        return [int(bit) for bit in binary]

    def convert_to_int(self, binary):
        binary_str = ''.join(str(bit) for bit in binary)
        return int(binary_str, 2)

    def adjust_weights(self, mostReleventKey):
        for entry in self.vector_db:
            
            binary_embedding = self.em.embeddingToBinary(entry["embedding"]) # may be better to just us a unquie identifier instead of a condensed embedding

            number_of_retrievals = entry["numberOfRetrievals"]
            number_of_retrievals_binary = self.convert_to_binary(number_of_retrievals)

            weight = int(entry["weight"]*10)
            weight = self.convert_to_binary(weight)

            input_to_agent = binary_embedding + number_of_retrievals_binary + weight

            if mostReleventKey == entry["input"]:
                self.most_recent_input = input_to_agent # somehow change to entry that has actually been retrieved
                print("found most relevent key in adjust weights")

            new_weight = self.convert_to_int(self.Agent.next_state(input_to_agent)) /10

            entry["weight"] = new_weight

            # Save the updated vector database
            self.vectorizer.save_cache()
            #
    def train_agent(self, type):

        weight = self.most_recent_input[14:18]
        weight = sum(weight)
        label = [0,0,0,0]
        print("old weight: ", weight)

        if type == "pos":
            for i in range(min(weight+1, 4)):
                label[i] = 1

            self.Agent.next_state(INPUT=self.most_recent_input, LABEL=label) # TODO use incremental learning
        else:
            for i in range(max(weight-1, 1)):
                label[i] = 1
            self.Agent.next_state(INPUT=self.most_recent_input, LABEL=label)

        
        print("trained : ", sum(label))
