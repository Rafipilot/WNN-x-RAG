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
        self.Agent.next_state(np.zeros(22), LABEL=[0,0,1,1])


    def convert_to_binary(self, interger):
        #print("converting : ", interger, " to binary")
        if interger == 0.2:
            binary = [0,0,0,0]
        elif interger == 0.4:
            binary = [0,0,0,1]
        elif interger == 0.6:
            binary = [0,0,1,1]
        elif interger == 0.8:
            binary = [0,1,1,1]
        elif interger == 1:
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
        elif integer >=4:
            binary = [1, 1, 1, 1]
        return binary

    

    def convert_to_int(self, binary_list):

        binary_list = binary_list.tolist()

        binary =[0,0,0,0]
        for i in range(sum(binary_list)):
            binary[i]= 1
        binary.reverse()
        if binary == [0,0,0,0]:
            integer = 0.2
        elif binary == [0,0,0,1]:
            integer = 0.4
        elif binary == [0,0,1,1]:
            integer = 0.6
        elif binary == [0,1,1,1]:
            integer = 0.8
        elif binary == [1,1,1,1]:
            integer  = 1
        return integer

    def adjust_weights(self, keys):
        self.most_recent_input = []
        for entry in self.vector_db:
            
            #binary_embedding = self.em.embeddingToBinary(entry["embedding"]) # may be better to just us a unquie identifier instead of a condensed embedding

            ID =  [int(bit) for bit in f"{entry["uniqueID"]:010b}"]

            number_of_retrievals = entry["numberOfRetrievals"]
            number_of_retrievals_binary = self.convert_int_to_binary(number_of_retrievals)

            weight = entry["weight"]
            weight = self.convert_to_binary(weight)

            numFailures = entry["numberFailures"]
            numFailuresBinary = self.convert_int_to_binary(numFailures)

            input_to_agent = ID + number_of_retrievals_binary + numFailuresBinary + weight

            for mostReleventKey in keys:
                # print("most relevent key ", mostReleventKey)
                # print("entry: ", entry["input"])
                if mostReleventKey == entry["input"]:
                    self.most_recent_input.append(input_to_agent) # entry that has actually been retrieved

            new_weight = self.convert_to_int(self.Agent.next_state(input_to_agent))
            self.Agent.reset_state()

            entry["weight"] = new_weight

            # Save the updated vector database
            self.vectorizer.save_cache()

            
    def train_agent(self, type, noResponse, keys, min_dist, i, actThresh):
        # print("keys: ", keys)
        # print("len keys", len(keys))
        # print("mri: ", self.most_recent_input)
        # print("len mri: ", len(self.most_recent_input))
        for most_relevent_key, recent_vec in zip(keys, self.most_recent_input):

            weighted = recent_vec[18:22]
            weight = sum(weighted)
            label = [0,0,0,0]
            if type == "pos":
                for i in range(min(weight+1, 4)):
                    label[i] = 1
                label.reverse()

            else:
                for i in range(max(weight-1, 1)):
                    label[i] = 1
                label.reverse()
                self.vectorizer.incrementNumberFailures(most_relevent_key)
            self.Agent.next_state(INPUT=recent_vec, LABEL=label)
            self.Agent.reset_state()
            
            actThresh.trainAgent(type, noResponse, min_dist, i)

            self.adjust_weights(most_relevent_key)  # Adjust weights after training the agent
            