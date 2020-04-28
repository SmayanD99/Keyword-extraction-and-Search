import PyPDF2 
import textract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import pandas as np


filename = 'enter the name of the file here'

pdfFileObj = open(filename,'rb')

pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

count = 0

text = ""
pageObj = pdfReader.getPage(0)
text = pageObj.extractText()

#This if statement exists to check if the above library returned words. It's done because PyPDF2 cannot read scanned files.
if text != "":
   text = text
#If the above returns as False,run OCR library textract to convert scanned/image based PDF files into text.
else:
   text = textract.process(fileurl, method='tesseract', language='eng')

#lowercasing eaxh word

text = text.encode('ascii','ignore').lower()

#word_tokenize() function breaks text phrases into individual words.
tokens = word_tokenize(text)
#We'll create a new list that contains punctuation we wish to clean.
punctuations = ['(',')',';',':','[',']',',']
stop_words = stopwords.words('english')
#We create a list comprehension that only returns a list of words that are NOT IN stop_words and NOT IN punctuations.
keywords = [word for word in tokens if not word in stop_words and not word in punctuations]

# #Alternative : Storing Keywords in Pandas Dataframe
# keywords = re.findall(r'[a-zA-Z]\w+',text)
# df = pd.DataFrame(list(set(keywords)),columns=['keywords'])

# Printing the keywords generated in a text file and then transferring
# for top keyword extraction using rake or TF-IDF Algorithms
# Yet to be done.