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
        if threshold == 0.15:
            binary = [0,0,0,0]
        elif threshold == 0.20:
            binary = [0,0,0,1]
        elif threshold == 0.25:
            binary = [0,0,1,1]
        elif threshold ==0.30:
            binary = [0,1,1,1]
        elif threshold == 0.35:
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
        # Calculate delta from current threshold to true min_dist
        delta = min((abs(self.threshold - min_dist) *2), 0.05)
        possible_values = [0.15, 0.20, 0.25, 0.30, 0.35]

        if type == "pos":
            print("Refining threshold (positive feedback)")
            target = self.threshold - (delta / 2)  # move halfway toward min_dist
        elif type == "neg" and not noResponse:
            print("Decreasing threshold (false positive)")
            target = self.threshold - delta
        elif type == "neg" and noResponse:
            print("Increasing threshold (false negative)")
            target = self.threshold + delta
        else:
            raise ValueError("ERROR: Invalid training input")


        closest = None
        min_diff = float("inf")
        for value in possible_values:
            diff = abs(value - target)
            if diff < min_diff:
                min_diff = diff
                closest = value

        self.threshold = closest
        label = self.convertThresholdToBinary(closest)

        print("Threshold label to closest:", closest)
        print("binary: ", label)

        self.Agent.next_state(self.previousInput, label)
        self.Agent.reset_state()




