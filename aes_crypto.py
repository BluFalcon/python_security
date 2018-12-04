# -*- coding: utf-8 -*-
"""

author: Kostas Vogiatzis
AM   :  2008030057  
mail :  kvogiat@gmail.com

info used :
    csrc.nist.gov/publications/fips/fips197/fips-197.pdf
    https://www.cs.utexas.edu/~mitra/honors/soln.html
    
"""

import random #gia ta random key
import numpy
import array

import copy


sample_txt = "eimai ena tuxaio keimeno gia na elegx8ei o algori8mos, btw i aek den 8a parei prwta8lima oute fetos..."


#
Sbox = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16)

InvSbox = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D)


Rcon = (
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39)


#eftiaxa egw stin stuxi ena IV gia to CBC mode 
IV = numpy.matrix('24 32 56 79;44 63 92 49;55 66 77 88;1 7 52 9')





class Crypto1:
   
    
    #den xrisimopoihsa tin os.urandom gt etsi k alliws den exei simasia edw
#personal note mipws valw auto
#http://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python/23728630#23728630

    def tuxaio_key(self,mege8os):
        key = random.randint(0, 9)
        for i in range(1,mege8os):
            key = key*10 + random.randint(0, 9)
        return str(key)
    #end tuxaio)key      



#Gia na kanoume split to text se kommatia
#source : http://stackoverflow.com/questions/13673060/split-string-into-strings-by-length
    def my_split_str(self,input_text,length):
      #version 7
      
        chunks, chunk_size = len(input_text), length
        m  = [ input_text[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
    
        l =  [numpy.reshape(bytearray(m[i]),len(m[i]) ) for i in range(0,len(m))] #auto epistrefei  to keimeno splitted se byte array

        return l
        
        
        
    #Zitaei kwdiko kai sumplirwnei oti leipei me \0 mexri na einai mod16=0
    def enter_password(self):
                
        usr_pass = raw_input('Enter password: ')
                
        k=len(usr_pass)%16
        
        if k != 0 :
           
            for i in range(0,16-k):
                usr_pass+=str("\0")
        
        #print(k,usr_pass,len(usr_pass))

        return usr_pass
    #end enter password
          
        
    
    def subBytes(self, k):
        for i in range(len(k)):
            k[i] = Sbox[k[i]]

    #end subBytes


    def invSubBytes(self, k):
        for i in range(len(k)):
            k[i] = InvSbox[k[i]]
    #end invSubBytes
    
    
    def subWords(self, s):
        for i in range(4):
            for j in range(4):
                s[i][j] = Sbox[s[i][j]]


    def invWord(self, s):
        for i in range(4):
            for j in range(4):
                s[i][j] = InvSbox[s[i][j]]

    
    #selida 17 prosoxi kanei shift toses 8eseis analoga me tin seira pou einai 
    def shiftRows(self,s):
        s[1][0], s[1][1], s[1][2], s[1][3] = s[1][1], s[1][2], s[1][3], s[1][0]
        s[2][0], s[2][1], s[2][2], s[2][3] = s[2][2], s[2][3], s[2][0], s[2][1]
        s[3][0], s[3][1], s[3][2], s[3][3] = s[3][3], s[3][0], s[3][1], s[3][2]
    #end shiftRows

    def invShiftRows(self,s):
        s[1][0], s[1][1], s[1][2], s[1][3] = s[1][3], s[1][0], s[1][1], s[1][2]
        s[2][0], s[2][1], s[2][2], s[2][3] = s[2][2], s[2][3], s[2][0], s[2][1]
        s[3][0], s[3][1], s[3][2], s[3][3] = s[3][1], s[3][2], s[3][3], s[3][0]
    #end invShiftRows
    
    
    
    def AddRoundKey(self,state,key):
        for i in range(4):
            for j in range(4):
                state[i][j] ^= key[i][j]


    #apo tin selida 18 tou manual antigramenes oi exiswseis
    def mixColumns(self,s):##einai la8os
         for i in range(4):
            s[0][i] = numpy.uint8((2*s[0][i])^(3*s[1][i])^s[2][i]^s[3][i])
            s[1][i] = numpy.uint8(s[0][i]^(2*s[1][i])^(3*s[2][i])^s[3][i])
            s[1][i] = numpy.uint8(s[0][i]^s[1][i]^(2*s[2][i])^(3*s[3][i]))
            s[3][i] = numpy.uint8((3*s[0][i])^s[1][i]^s[2][i]^(2*s[3][i]))


   #apo tin selida 23 tou manual antigramenes oi exiswseis
    def InvMixColumns(self,s):##einai la8os
         for i in range(4):
            s[0][i] = numpy.uint8((14*s[0][i])^(11*s[1][i])^(13*s[2][i])^(9*s[3][i]))
            s[1][i] = numpy.uint8((9*s[0][i])^(14*s[1][i])^(11*s[2][i])^(13*s[3][i]))
            s[2][i] = numpy.uint8((13*s[0][i])^(9*s[1][i])^(14*s[2][i])^(11*s[3][i]))
            s[3][i] = numpy.uint8((11*s[0][i])^(13*s[1][i])^(9*s[2][i])^(14*s[3][i]))
    
    #apo sel 14
    #typeof key   Length key   Block Size    Num of Rounds
    #aes-128        Nk=4        Nb=4           Nr=10
    #aes-192        Nk=6        Nb=4           Nr=12
    #aes-256        Nk=8        Nb=4           Nr=14


    #routina pou paragei 44 keys opws perigrafetai stis se 19-20 tou fips-197.pdf
    def keyExpansion(self,key,Nk):
        #edw paragontai ta kleidia akolou8isa tin perigrafi tou manual
        Nb=4
        
        if Nk==4:
            Nr=10
        elif Nk==6:
            Nr=12
        elif Nk==8:
            Nr=14
        else:
            print("ERROR in keyExpansion")
        
        w = [None]*Nb*(Nr+1);
        
        key = self.my_split_str(str(my_key),4) 
                
        i=0
        while(i<Nk):
            #w[i]=[int(key[4*i]),int(key[4*i+1]),int(key[4*i+2]),int(key[4*i+3] )]
            w[i]=[key[i][0],key[i][1],key[i][2],key[i][3] ]
            i+=1
            
        i=Nk
        
                
        while(i <Nb*(Nr+1)):
            temp = w[i-1]
        
            if(i%Nk==0):
                temp = [temp[1],temp[2],temp[3],temp[0]]
                self.subBytes(temp)
                temp[0] ^= Rcon[i/Nk]  ##Rcon[i], contains the values given by [xi-1,{00},{00},{00}] ype8esa oti einai to temp[0] an vgainei kati la8os 8a ennoei to temp[3] :D
            elif(Nk >6) and (i%Nk == 4):
                temp = [temp[0],temp[1],temp[2],temp[3]]
                self.subBytes(temp)
            else:
               temp
               
            w[i]=temp # den afinei na ginei assigne se None.. ara auto xreiazetai gia arxikopoihsh
                
            w[i][0]=w[i-Nk][0] ^ temp[0]
            w[i][1]=w[i-Nk][1] ^ temp[1]
            w[i][2]=w[i-Nk][2] ^ temp[2]
            w[i][3]=w[i-Nk][3] ^ temp[3]
                
            i+=1
                                    
        return  w
            
        
    
    
    def myEncrypt(self,plain_txt,mode,the_key,Nk):
        #only 2 mods ECB & CBC https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation
        Nb=4
        
        if Nk==4:
            Nr=10
        elif Nk==6:
            Nr=12
        elif Nk==8:
            Nr=14
        else:
            print("ERROR in myEncryption")
       
        k=len(plain_txt)%128
        if k!=0 :
            for i in range(0,128-k):
                plain_txt+=str("\0")                
        
        
        #Kovoume to keimeno se kommatia twn 16byte=128bit
        txt_in_pieces = self.my_split_str(plain_txt,16)
        key_array = self.keyExpansion(the_key,Nk)
        
        chipers = [None]*len(txt_in_pieces)    
        
        if mode == "ECB":
            for i in range(len(txt_in_pieces)): ## gia ka8e kommati
                state = numpy.reshape(txt_in_pieces[i],(4,4))    
                self.AddRoundKey(state , key_array[0: Nb]) 
                
                for r in range(1,Nr+1): #ta 10 stadia tou ka8e kommatiou
                    self.subWords(state)                    
                    self.shiftRows(state) 
                    #self.mixColumns(state) 
                    self.AddRoundKey(state, key_array[r*Nb: ((r+1)*Nb) + 1])
                
                chipers[i] = state  

        elif mode == "CBC":
            
            for i in range(len(txt_in_pieces)):
                state = numpy.reshape(txt_in_pieces[i],(4,4))    
                
                if(i==0):  #edw kanw to XOR me to Initialization vector 
                    state = numpy.array( numpy.uint8(state^IV) )
                    #i apo panw grammi kanei auto --> state ^= IV
                else:
                    state ^=chipers[i-1]

                    #print("b",chipers[i-1])
                
                self.AddRoundKey(state, key_array[0: Nb])                 
                for r in range(1,Nr+1):
                    self.subWords(state)
                    self.shiftRows(state) 
                    #self.mixColumns(state) 
                    self.AddRoundKey(state, key_array[r*Nb: ((r+1)*Nb) + 1])
                
                chipers[i] = state  
        
        else:
            print("error 2 in mode at myEncrypt")

        
        return chipers 
    #end myencrypt
           

    
    def myDecrypt(self,chipers,mode,cr_key,Nk):
      
        Nb=4
        
        if Nk==4:
            Nr=10
        elif Nk==6:
            Nr=12
        elif Nk==8:
            Nr=14
        else:
            print("ERROR in myDecrypt")
       
        txt = [None]*len(chipers)    
#        txt = chipers[:]    
        
        key_array = self.keyExpansion(cr_key,Nk)
        
        #for i in range(1,Nr-1):
         #   self.InvMixColumns(key_array[i*Nb: (i+1)*Nb]) 
                    
        
        
        if mode == "ECB":
            for i in range(len(chipers)): ## gia ka8e kommati
                state = chipers[i]   
                self.AddRoundKey(state , key_array[Nr*Nb: (Nr+1)*Nb]) 
                
                for r in range(Nr-1,-1,-1): #ta 10 stadia tou ka8e kommatiou
                    self.invWord(state)                    
                    self.invShiftRows(state) 
                    #self.InvMixColumns(state) 
                    self.AddRoundKey(state, key_array[r*Nb: ((r+1)*Nb) + 1])
                    
                txt[i] = state  

        elif mode == "CBC":
            
            d =  copy.deepcopy(chipers)
            
            for i in range(len(txt)):
                
                state = d[i]   
                
                self.AddRoundKey(state, key_array[Nr*Nb: (Nr+1)*Nb]) 
                for r in range(Nr-1,-1,-1):
                    self.invWord(state)
                    self.invShiftRows(state) 
                    #self.InvMixColumns(state) 
                    self.AddRoundKey(state, key_array[r*Nb: ((r+1)*Nb) + 1])
                    
                    
                if(i==0):  #edw kanw to XOR me to Initialization vector 
                    state = numpy.array( numpy.uint8(state^IV) )
                    #i apo panw grammi kanei auto --> state ^= IV
                else:
                    state ^= chipers[i-1]
                    #print(chipers[i-1])
               
                txt[i] = state  


        else:
            print("error 2 in mode at myDecrypt")



        txt_fin = [None]*len(txt)*4

        for i in range(len(txt)):
            for j in range(4):
                txt_fin[i*4+j] = "".join(map(chr, txt[i][j]))
        
        #txt_fin = "".join(txt_fin)
        
        return txt_fin

        #end mydecrypt
   
    
       

class Crypto2:
    #https://www.cs.utexas.edu/~mitra/honors/soln.html
    
    def createPrime(self,lower,upper):
        for num in range(lower,upper + 1):
   # prime numbers are greater than 1
            if num > 1:
               for i in range(2,num):
                   if (num % i) == 0:
                       break
               else:
                   print(num)
                
        return
        
     
############################################################        

a = Crypto1()

Nk = 4
my_key = a.tuxaio_key(16)


'''

#to key pou dinei to manual key = str('2b7e151628aed2a6abf7158809cf4f3c')


        

'''

#r = a.keyExpansion(my_key,Nk)
#t =a.myEncrypt(sample_txt,"CBC",my_key,4)
#
t = a.myEncrypt(sample_txt,"CBC",my_key,4)

t2 = copy.deepcopy(t)


print("1",t)
#
k=a.myDecrypt(t,"CBC",my_key,4)
#print("2",t)

#plain_txt  = sample_txt
#a.mixColumns(r)
#a.InvMixColumns(r)




#EKDOSI GIA NA MHN XA8EI I PRO8ESMIA






