### This script dynamically adjust the weights of items in the vector database based on their relevance to the input embedding.

import ao_core as ao
from config import openai_key
import ao_embeddings.binaryEmbeddings as be
import numpy as np
import warnings

class weightController:
    def __init__(self, vectorizer):

        self.vectorizer = vectorizer
        self.vector_db = vectorizer.cache
        self.Arch = ao.Arch(arch_i=[10,4,4, 20], arch_z=[20]) # Input is condensed embedding, number of retrivals, current weight. Output is the next weight # TODO add a unique identifierS
        self.Agent = ao.Agent(Arch=self.Arch)
        self.em = be.binaryEmbeddings(openai_api_key=openai_key, numberBinaryDigits=10)
        Label = np.zeros(20)
        Label[0:16] = 1
        np.flip(Label)
        self.Agent.next_state(np.zeros(38), LABEL=Label)


    def convert_to_binary(self, interger):
        interger = round(interger,2)
        num_ones = int(interger *20)
        binary = np.zeros(20)
        binary[0:num_ones] = 1
        np.flip(binary)
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
        else:
            print("error: ", integer)
        return binary

    

    def convert_to_int(self, binary_list):

        binary_list = binary_list.tolist()
        
        num_ones = sum(binary_list)

        integer = num_ones/20
            
        return integer

    def adjust_weights(self):
        self.most_recent_inputs = []
        for entry in self.vector_db:
            
            #binary_embedding = self.em.embeddingToBinary(entry["embedding"]) # may be better to just us a unquie identifier instead of a condensed embedding

            ID =  [int(bit) for bit in f"{entry["uniqueID"]:010b}"]

            number_of_retrievals = entry["numberOfRetrievals"]
            number_of_retrievals_binary = self.convert_int_to_binary(number_of_retrievals)

            weight = entry["weight"]
            weight = self.convert_to_binary(weight).tolist()

            numFailures = entry["numberFailures"]
            numFailuresBinary = self.convert_int_to_binary(numFailures)

            input_to_agent = ID + number_of_retrievals_binary + numFailuresBinary + weight

            self.most_recent_inputs.append(input_to_agent)

            new_weight = self.convert_to_int(self.Agent.next_state(input_to_agent))
            self.Agent.reset_state()

            entry["weight"] = new_weight

            # Save the updated vector database
            self.vectorizer.save_cache()

            
    def train_agent(self, type, noResponse, key,  min_dist, index, actThresh):
        # print("keys: ", keys)
        # print("len keys", len(keys))
        # print("mri: ", self.most_recent_input)
        # print("len mri: ", len(self.most_recent_input))
        #if noResponse == False:
        if noResponse == False:
            recent_vec = None
            for i, value in enumerate(self.vector_db):
                if value["input"] == key:
                    recent_vec = self.most_recent_inputs[i]
                    break
            if not recent_vec:
                warnings.warn("No recent vec ERROR")
                print("vect db: ", [item["input"] for item in self.vector_db])
                print("key: ", key)
    
            weighted = recent_vec[-20:]
            print("Weight: ", weighted)
            weight = int(sum(weighted))
            label = np.zeros(20)
            if type == "pos":
                for i in range(min(weight+4, 20)):
                    label[i] = 1
                np.flip(label)

            else:
                for i in range(max(weight-1, 1)):
                    label[i] = 1
                np.flip(label)
                self.vectorizer.incrementNumberFailures(key)
                print("negative training: ", weighted, " to ", label)
            self.Agent.next_state(INPUT=recent_vec, LABEL=label, unsequenced=False)
            self.Agent.reset_state()
            
            actThresh.trainAgent(type, noResponse, min_dist, index)

            
        elif noResponse == True and type == "pos":
            for INPUT in self.most_recent_inputs:
                self.Agent.next_state(INPUT, Cpos= True, unsequenced=True)
                self.Agent.reset_state()
        elif noResponse == True and type == "neg":
            for INPUT, entry in zip(self.most_recent_inputs, self.vector_db):
                # Increase the weight of everything incrementally
                target_weight = min(entry["weight"]+0.05, 1)
                target_weight_binary = self.convert_to_binary(target_weight)

                self.Agent.next_state(INPUT, target_weight_binary)
        else:
            warnings.warn("Invalid Response raising error")
            raise ValueError 
            
        self.adjust_weights()  # Adjust weights after training the agent


    def increase_weight(self, answer):
        for entry in self.vector_db:
            if answer in entry["input"]:
                ID =  [int(bit) for bit in f"{entry["uniqueID"]:010b}"]

                number_of_retrievals = entry["numberOfRetrievals"]
                number_of_retrievals_binary = self.convert_int_to_binary(number_of_retrievals)

                weight = entry["weight"]
                weight = self.convert_to_binary(weight).tolist()

                numFailures = entry["numberFailures"]
                numFailuresBinary = self.convert_int_to_binary(numFailures)

                input_to_agent = ID + number_of_retrievals_binary + numFailuresBinary + weight 

                label = np.zeros(20)
                target = int(min((sum(weight)+1),20))
                label[0:target]=1
                self.Agent.next_state(input_to_agent, label,unsequenced=True)
                self.Agent.reset_state()

    def reset_weights(self):
        for entry in self.vector_db:
            entry["weight"] = 0.8
        self.vectorizer.save_cache()