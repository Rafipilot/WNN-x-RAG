## AO WNN agent to adjust the threshold actively based on the embeddings being compared 

import ao_core as ao
from config import openai_key
import ao_embeddings.binaryEmbeddings as be
import numpy as np

class activeThreshold:
    def __init__(self):
        self.threshold = 0.25
        self.Arch = ao.Arch(arch_i=[510], arch_z=[4]) # Input is condensed embedding, number of retrivals, current weight. Output is the next weight # TODO add a unique identifierS
        self.Agent = ao.Agent(Arch=self.Arch)
        self.em = be.binaryEmbeddings(openai_api_key=openai_key, numberBinaryDigits=500)

        self.Agent.next_state(INPUT=np.zeros(510), LABEL=[0,0,0,1])

    def convertThresholdToBinary(self, threshold):
        threshold = round(threshold,2)
        if threshold > 0.2:
            binary = [0,0,0,0]
        elif threshold == 0.25:
            binary = [0,0,0,1]
        elif threshold == 0.30:
            binary = [0,0,1,1]
        elif threshold ==0.35:
            binary = [0,1,1,1]
        elif threshold < 0.40:
            binary = [1,1,1,1]
        return binary
    
    def convertBinaryToThreshold(self, binary):
        binary = binary.tolist()
        if binary == [0, 0, 0, 0]:
            threshold = 0.2
        elif binary == [0, 0, 0, 1]:
            threshold = 0.25
        elif binary == [0, 0, 1, 1]:
            threshold = 0.30
        elif binary == [0, 1, 1, 1]:
            threshold = 0.35
        elif binary ==[1,1,1,1]:
            threshold = 0.4 
        else:
            print("unknown binary: ", binary)
        return threshold

    def adjustThreshold(self, entry):

        binary_embedding = self.em.embeddingToBinary(entry["embedding"])
        ID =  [int(bit) for bit in f"{entry["uniqueID"]:010b}"]

        input_to_agent = binary_embedding+ID
        output = self.Agent.next_state(input_to_agent)
        print("thrshold moved from: ", self.threshold)
        self.threshold = self.convertBinaryToThreshold(output)

        self.previousInput= input_to_agent

        print("to : ", self.threshold)

        return self.threshold

    def trainAgent(self, type):
        print("type of t for thr: ", type)
        if type == "pos":
            target = self.threshold -0.05
        else:
            target = self.threshold + 0.05
        print("target: ", target)
        label = self.convertThresholdToBinary(target)

        self.Agent.next_state(self.previousInput, label)

        print("Trained threshold from", self.threshold, "to ", label)


