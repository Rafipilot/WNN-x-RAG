### This script dynamically adjust the weights of items in the vector database based on their relevance to the input embedding.

import ao_core as ao
from config import openai_key
import ao_embeddings.binaryEmbeddings as be
import numpy as np
import warnings

class weightController:
    def __init__(self, vectorizer):

        self.vectorizer = vectorizer
        self.vector_db = vectorizer.vectorDB
        self.Arch = ao.Arch(arch_i=[10,4,4], arch_z=[20]) # Input is condensed embedding, number of retrivals, current weight. Output is the next weight # TODO add a unique identifierS
        self.Agent = ao.Agent(Arch=self.Arch)
        self.em = be.binaryEmbeddings(openai_api_key=openai_key, numberBinaryDigits=10)
        Label = np.zeros(20)
        Label[0:16] = 1 # -> 0.8
        Label = np.flip(Label)
        self.Agent.next_state(np.zeros(18), LABEL=Label)
        self.adjust_weights()



    def convert_to_binary(self, interger):
        interger = round(interger,2)
        num_ones = int(interger *20)
        binary = np.zeros(20)
        binary[0:num_ones] = 1
        binary = np.flip(binary)
        return binary
    
    def convert_int_to_binary(self, integer):

        if integer <= 10:
            binary = [0, 0, 0, 0]
        elif integer <=25:
            binary = [0, 0, 0, 1]
        elif integer <=40:
            binary = [0, 0, 1, 1]
        elif integer <=60:
            binary = [0, 1, 1, 1]
        elif integer >=61:
            binary = [1, 1, 1, 1]
        else:
            print("error: ", integer)
        return binary

    

    def convert_to_int(self, binary_list):

        binary_list = binary_list.tolist()
        
        num_ones = sum(binary_list)

        integer = num_ones/20
            
        return integer
    
    def create_input_to_agent(self, entry):
        ID =  [int(bit) for bit in f"{entry['uniqueID']:010b}"]

        number_of_retrievals = entry["numberOfRetrievals"]
        number_of_retrievals_binary = self.convert_int_to_binary(number_of_retrievals)

        numFailures = entry["numberFailures"]
        numFailuresBinary = self.convert_int_to_binary(numFailures)

        input_to_agent = ID + number_of_retrievals_binary + numFailuresBinary

        return input_to_agent

    def adjust_weights(self):
        self.most_recent_inputs = []
        for entry in self.vector_db:
            
            #binary_embedding = self.em.embeddingToBinary(entry["embedding"]) # may be better to just us a unquie identifier instead of a condensed embedding

            input_to_agent = self.create_input_to_agent(entry)
            self.most_recent_inputs.append(input_to_agent)

            new_weight = self.convert_to_int(self.Agent.next_state(input_to_agent, unsequenced=True))
            self.Agent.reset_state()

            entry["weight"] = new_weight

            # Save the updated vector database
            self.vectorizer.save_vectorDB()

            
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
                for i in range(min(weight+8, 20)):
                    label[i] = 1
            else:
                for i in range(max(weight-1, 1)):
                    label[i] = 1
                
                self.vectorizer.incrementNumberFailures(key)
                print("negative training: ", weighted, " to ", label)
            label = np.flip(label)
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
                target_weight = min(entry["weight"]+0.1, 1)
                target_weight_binary = self.convert_to_binary(target_weight)

                self.Agent.next_state(INPUT, target_weight_binary, unsequenced=True)
                self.Agent.reset_state()
        else:
            warnings.warn("Invalid Response raising error")
            raise ValueError 



    def increase_target_weight(self, answer):
        for entry in self.vector_db:
            if answer in entry["input"]:
                input_to_agent = self.create_input_to_agent(entry)
                weight = entry["weight"] 

                label = np.zeros(20)
                target = int(min((int(weight*20)+12),20))
                label[0:target]=1
                label = np.flip(label)
                self.Agent.next_state(input_to_agent, label,unsequenced=True)
                self.Agent.reset_state()
        

    def reset_weights(self):
        for entry in self.vector_db:
            entry["weight"] = 0.9
        self.vectorizer.save_vectorDB()