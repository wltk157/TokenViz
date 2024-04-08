import os.path

import numpy
import numpy as np

import re

from tqdm import tqdm
import math



def tokenize_code(code):
    
    
    pattern = r'\b\w+\b|[-+*/=<>()[\]{};]|[\n\t]'
    tokens = re.findall(pattern, str(code))
    return tokens

def add_indentation(java_code, num_tabs=1):
    
    indented_code = re.sub(r'(^|\n)', r'\1' + '\t' * num_tabs, java_code)
    return indented_code

class TokenVisDataset:

    def __init__(self,embed,vocab,row,col,blk_width,blk_height,variable=True,new_line=True,folder='IR'):
        
        self.embed_weight=embed.weight.data 
        self.vocab=vocab 
        self.row=row 
        self.col=col 
        self.blk_width=blk_width 
        self.blk_height=blk_height 

        self.variable=variable 
        self.newline=new_line 
        self.folder = folder  



    def toDict(self):
        
        return {'row': self.row, 'col': self.col, 'blk_width': self.blk_width, 'data': self.data}

    def append(self, id, npydata):
        self.data.update({id: npydata})

    def token2block(self, token):
        
        
        
        vec = self.embed_weight[self.vocab[token]]
        return vec.reshape(self.blk_height,self.blk_width).cpu().numpy()



        
    def visualize(self,code):
        
        

        empty_block = numpy.zeros([self.blk_height, self.blk_width])

        if not self.newline:
            tokens = tokenize_code(str(code)) 
            block_tokens =  [self.token2block(token)for token in tokens]

            if self.variable:
                
                num_tokens_row=num_tokens_col=math.ceil(math.sqrt(len(tokens)))
            else:
                
                num_tokens_row=self.row
                num_tokens_col=self.col

                
                block_tokens=block_tokens[:num_tokens_row*num_tokens_col]


            
            
            block_tokens = block_tokens + [empty_block] * (num_tokens_row * num_tokens_col - len(block_tokens))

            
            block_lines = [block_tokens[i: i + num_tokens_col] for i in range(0, len(block_tokens), num_tokens_col)]
            blk_rows = [np.concatenate(line, axis=1) for line in block_lines]
            codeimg = np.concatenate(blk_rows, axis=0)


        elif self.variable:
            

            max_width=0
            code_lines = [tokenize_code(line) for line in str(code).split('\n')]
            block_lines = [[self.token2block(token) for token in line] for line in code_lines]

            
            for line in block_lines:
                if max_width<len(line):
                    max_width=len(line)


            
            block_lines = [line + [empty_block] * (max_width - len(line))
                           if len(line) < max_width else line
                           for line in block_lines]

            
            blk_rows = [np.concatenate(line, axis=1) for line in block_lines]
            codeimg = np.concatenate(blk_rows, axis=0)



        else:
            
            
            code_lines = [tokenize_code(line) for line in code.split('\n')]

            
            

            
            block_lines = [[self.token2block(token) for token in line] for line in code_lines]

            
            block_lines = [line[i:i + self.col] for line in block_lines for i in range(0, len(line), self.col)]

            

            block_lines = [line + [empty_block] * (self.col - len(line))
                           if len(line) < self.col else line
                           for line in block_lines[:self.row]]

            
            blk_rows = [np.concatenate(line, axis=1) for line in block_lines]
            codeimg = np.concatenate(blk_rows, axis=0)
            codeimg = np.pad(codeimg, [(0, self.row * self.blk_width - codeimg.shape[0]), (0, 0)], 'constant',
                             constant_values=0)


        
        codeimg=np.expand_dims(codeimg,0)

        return codeimg
        
        
        




    def build(self, inputs):
        
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)

        print("Creating visual representation...")
        for id, code in tqdm(inputs):
            npydata = self.visualize(str(code))
            file_path = self.folder + "/" + str(id) + ".npy"
            np.save(file_path, npydata)



