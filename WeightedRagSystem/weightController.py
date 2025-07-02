### This script dynamically adjust the weights of items in the vector database based on their relevance to the input embedding.

import ao_core as ao
from config import openai_key
import ao_embeddings.binaryEmbeddings as be

import numpy as np

class weightController:
    def __init__(self, vectorizer):

        self.vectorizer = vectorizer
        self.vector_db = vectorizer.cache
        self.Arch = ao.Arch(arch_i=[10,4,4, 4], arch_z=[4]) # Input is condensed embedding, number of retrivals, current weight. Output is the next weight # TODO add a unique identifierS
        self.Agent = ao.Agent(Arch=self.Arch)
        self.em = be.binaryEmbeddings(openai_api_key=openai_key, numberBinaryDigits=10)


    def convert_to_binary(self, interger):
        if interger == 0.2:
            binary = [0,0,0,0]
        elif interger == 0.4:
            binary = [0,0,0,1]
        elif interger == 0.6:
            binary = [0,0,1,1]
        elif interger == 0.8:
            binary = [0,1,1,1]
        else:
            binary = [1,1,1,1]
        return binary
    
    def convert_int_to_binary(self, integer):
        if integer == 0:
            binary = [0, 0, 0, 0]
        elif integer == 1:
            binary = [0, 0, 0, 1]
        elif integer == 2:
            binary = [0, 0, 1, 1]
        elif integer == 3:
            binary = [0, 1, 1, 1]
        else:
            binary = [1, 1, 1, 1]
        return binary

    

    def convert_to_int(self, binary):
        try:
            binary = binary.tolist()
        except Exception as e:
            pass
        if binary == [0,0,0,0]:
            integer = 0.2
        elif binary == [0,0,0,1]:
            integer = 0.4
        elif binary == [0,0,1,1]:
            integer = 0.6
        elif binary == [0,1,1,1]:
            integer = 0.8
        else:
            integer  = 1
        return integer

    def adjust_weights(self, mostReleventKey):
        for entry in self.vector_db:
            
            binary_embedding = self.em.embeddingToBinary(entry["embedding"]) # may be better to just us a unquie identifier instead of a condensed embedding

            ID =  [int(bit) for bit in f"{entry["uniqueID"]:010b}"]

            number_of_retrievals = entry["numberOfRetrievals"]
            number_of_retrievals_binary = self.convert_to_binary(number_of_retrievals)

            weight = entry["weight"]
            weight = self.convert_to_binary(weight)

            numFailures = entry["numberFailures"]
            numFailuresBinary = self.convert_int_to_binary(numFailures)

            input_to_agent = ID + number_of_retrievals_binary + numFailuresBinary + weight

            if mostReleventKey == entry["input"]:
                self.most_recent_input = input_to_agent # entry that has actually been retrieved

            new_weight = self.convert_to_int(self.Agent.next_state(input_to_agent))

            entry["weight"] = new_weight

            # Save the updated vector database
            self.vectorizer.save_cache()

            
    def train_agent(self, type, most_relevant_key, actThresh):

        weighted = self.most_recent_input[18:22]
        print("from train: ",weighted)
        weight = sum(weighted)
        label = [0,0,0,0]
        print("old weight: ", self.convert_to_int(weighted))

        if type == "pos":
            for i in range(min(weight+1, 4)):
                label[i] = 1
            label.reverse()

        else:
            for i in range(max(weight-1, 1)):
                label[i] = 1
            label.reverse()
            self.vectorizer.incrementNumberFailures(most_relevant_key)

        self.Agent.next_state(INPUT=self.most_recent_input, LABEL=label)
        actThresh.trainAgent(type)

        
        print("trained : ", self.convert_to_int(label))

        self.adjust_weights(most_relevant_key)  # Adjust weights after training the agent
        
