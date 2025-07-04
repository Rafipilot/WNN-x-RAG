## AO WNN agent to adjust the threshold actively based on the embeddings being compared 

import ao_core as ao
from config import openai_key
import ao_embeddings.binaryEmbeddings as be
import numpy as np

class activeThreshold:
    def __init__(self):
        self.threshold = 0.25
        self.Arch = ao.Arch(arch_i=[500, 500, 10], arch_z=[4]) # Input is condensed embedding, number of retrivals, current weight. Output is the next weight # TODO add a unique identifierS
        self.Agent = ao.Agent(Arch=self.Arch)
        self.em = be.binaryEmbeddings(openai_api_key=openai_key, numberBinaryDigits=500)

        self.Agent.next_state(INPUT=np.zeros(1010), LABEL=[0,0,1,1]) # init train with valid binary output

    def convertThresholdToBinary(self, threshold):
        threshold = round(threshold,2)
        if threshold > 0.15:
            binary = [0,0,0,0]
        elif threshold == 0.20:
            binary = [0,0,0,1]
        elif threshold == 0.25:
            binary = [0,0,1,1]
        elif threshold ==0.30:
            binary = [0,1,1,1]
        elif threshold < 0.35:
            binary = [1,1,1,1]
        return binary
    
    def convertBinaryToThreshold(self, inb):
        inb= inb.tolist()
        binary = [0,0,0,0]
        for i in range(sum(inb)):
            binary[i] = 1
        binary.reverse()
        if binary == [0, 0, 0, 0]:
            threshold = 0.15
        elif binary == [0, 0, 0, 1]:
            threshold = 0.20
        elif binary == [0, 0, 1, 1]:
            threshold = 0.25
        elif binary == [0, 1, 1, 1]:
            threshold = 0.30
        elif binary ==[1,1,1,1]:
            threshold = 0.35 
        else:
            print("unknown binary: ", binary)
        return threshold

    def adjustThreshold(self, entry, userInputEmbedding):



        DB_embedding_binary = self.em.embeddingToBinary(entry["embedding"])
        userInputEmbeddingBinary = self.em.embeddingToBinary(userInputEmbedding)
        ID =  [int(bit) for bit in f"{entry["uniqueID"]:010b}"]

        input_to_agent = DB_embedding_binary+userInputEmbeddingBinary+ID
        output = self.Agent.next_state(input_to_agent)
        self.Agent.reset_state()
        self.threshold = self.convertBinaryToThreshold(output)

        self.previousInput= input_to_agent

        return self.threshold

    def trainAgent(self, type, noResponse, min_dist):
        delta = abs(self.threshold-min_dist)
        target_delta = delta/2
        possible_values = [0.15, 0.20, 0.25, 0.30, 0.35]

        if type == "pos":
            # Try to refine the threshold closer to the actual min_dist
            target = (self.threshold + min_dist) / 2
            print("Refining threshold (positive feedback)")

        elif type == "neg" and noResponse == False:
            print("Decreasing threshold")
            target = self.threshold - target_delta
        elif type == "neg" and noResponse == True:
            target = self.threshold + target_delta
            print("Increasing threshold")
        
        else:
            raise ValueError("ERROR!")

        # get threshold to one of the possible values
        closest = None
        closest_dist = float("inf")
        for item in possible_values:
            delta = abs(self.threshold - item)
            if delta < closest_dist:
                closest_dist = delta
                closest = item
        self.threshold = closest
        label = self.convertThresholdToBinary(target)

        self.Agent.next_state(self.previousInput, label)
        self.Agent.reset_state()




