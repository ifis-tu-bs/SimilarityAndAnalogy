'''
Created on 08.12.2014

@author: Christoph
'''

class Challenge(object):
    """ This class stores a SAT challenge and its answer options """ 

    def __init__(self):
        """ Basic constructor """
        self.options =  []
     
     
    def addOption(self,rawline):
        """ Add a new option by reading a raw line from the SAT file 
        rawline : str String as read from SAT file """
        splits=rawline.strip().split(" ")
        self.options.append(splits)
               
    
    def setCorrectIndex(self, index):
        """ Sets the index of the correct option, and stores it as a number.
        rawline : str Strinf as read from SAT file """
        self.correctIndex=ord(index[0])-96
     
        
    def __str__(self): 
        """ Simply converts this challenge into a human readable string """
        result="\nChallenge:\n"
        for option in self.options:
            result+="["
            for part in option:
                result+=part+" "
            result+="]"
        result+=" >> "+str(self.correctIndex)    
        return result
        
