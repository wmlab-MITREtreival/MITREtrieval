import pandas as pd
from query_list import *
import re
import json

def get_authentication():
    with open('Data/neo4j_info.json') as f:
        neo4j_info = json.load(f)
    f.close()
    return neo4j_info
			
def IOC_parser(test_string): #this is for parsing IOC from a report
	'''
	input:sentence: string a document
	'''
	IOC_IP=re.findall(r'(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})|(?:\d{1,3}\[.]\d{1,3}\[.]\d{1,3}\[.]\d{1,3})',test_string)
	print(IOC_IP)
	IOC_CVE=re.findall(r'\b(?:CVE\-[0-9]{4}\-[0-9]{4,6})\b',test_string)
	print(IOC_CVE)
	IOC_email=re.findall(r'\b(?:[a-z][_a-z0-9-.]+@[a-z0-9-]+\.[a-z]+)\b|\b(?:[a-z][_a-z0-9-[.]]+@[a-z0-9-]+\[.][a-z]+)\b',test_string)
	print(IOC_email)
	IOC_MD5=re.findall(r'\b(?:[a-f0-9]{32}|[A-F0-9]{32})\b',test_string)
	print(IOC_MD5)
	IOC_registry=re.findall(r'\b(?:(HKLM|HKCU)([\\A-Za-z0-9-_]+))\b',test_string)
	print(IOC_registry)
	IOC_SHA1=re.findall(r'\b(?:[a-f0-9]{40}|[A-F0-9]{40}|[0-9a-f]{40})\b',test_string)
	print(IOC_SHA1)
	IOC_SHA256=re.findall(r'\b([a-f0-9]{64}|[A-F0-9]{64})\b',test_string)
	print(IOC_SHA256)
	IOC_URL=re.findall(r'[a-z0-9]+(?:\.[a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj| Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)',test_string)
	print(IOC_URL)
	IOC_hash=re.findall(r'\b[a-fA-F\d]{32}\b|\b[a-fA-F\d]{40}\b|\b[a-fA-F\d]{64}\b',test_string)
	print(IOC_hash)
	#print(IOC_IP, IOC_CVE, IOC_email, IOC_MD5, IOC_registry, IOC_SHA1, IOC_SHA256, IOC_URL,IOC_vul, IOC_file,IOC_hash)
	return IOC_IP, IOC_CVE, IOC_email, IOC_MD5, IOC_registry, IOC_SHA1, IOC_SHA256, IOC_URL,IOC_vul, IOC_file,IOC_hash
def test(test_string):
	IOC_IP=re.findall(r'(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})|(?:\d{1,3}\[.]\d{1,3}\[.]\d{1,3}\[.]\d{1,3})',test_string)
	print(IOC_IP)
	IOC_CVE=re.findall(r'\b(?:CVE\-[0-9]{4}\-[0-9]{4,6})\b',test_string)
	print(IOC_CVE)
	IOC_email=re.findall(r'\b(?:[a-z][_a-z0-9-.]+@[a-z0-9-]+\.[a-z]+)\b|\b(?:[a-z][_a-z0-9-[.]]+@[a-z0-9-]+\[.][a-z]+)\b',test_string)
	print(IOC_email)
	IOC_MD5=re.findall(r'\b(?:[a-f0-9]{32}|[A-F0-9]{32})\b',test_string)
	print(IOC_MD5)
	IOC_registry=re.findall(r'\b(?:(HKLM|HKCU)([\\A-Za-z0-9-_]+))\b',test_string)
	print(IOC_registry)
	IOC_SHA1=re.findall(r'\b(?:[a-f0-9]{40}|[A-F0-9]{40}|[0-9a-f]{40})\b',test_string)
	print(IOC_SHA1)
	IOC_SHA256=re.findall(r'\b([a-f0-9]{64}|[A-F0-9]{64})\b',test_string)
	print(IOC_SHA256)
	IOC_URL=re.findall(r'[a-z0-9]+(?:\.[a-z0-9]+)*\[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj| Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)',test_string)
	print(IOC_URL)
	IOC_hash=re.findall(r'\b[a-fA-F\d]{32}\b|\b[a-fA-F\d]{40}\b|\b[a-fA-F\d]{64}\b',test_string)
	print(IOC_hash)
	IOC_FILE=re.findall(r'(?:[A-Za-z0-9]+\.(?:jpg|JPG|gif|GIF|doc|DOC|pdf|PDF|cpp|c))',test_string)
	print(IOC_FILE)
test("abc.jpg")

def string_found(string1, string2):
   if re.search(r"\b" + re.escape(string1) + r"\b", string2):
      return True
   return False
def search_nick(lists,text):
	for iter in lists:
		if string_found(iter.lower(),text):
			return True
	return False
'''
 * group_software_parser()-Delete brackets in CTI report
 * @text: Text in CTI report
'''
def group_software_parser(text):
	#greeter = Ontology("bolt://140.115.54.90:10096", "neo4j", "wmlab")
	neo4j_info = get_authentication()
	greeter = Ontology(neo4j_info["url"], neo4j_info["account"], neo4j_info["password"])
	group = greeter.query_all_group_name()
	software=greeter.query_all_software_name()

	#print(software)
	all_soft=[]
	all_group=[]
	for soft_iter in software:
		if string_found(soft_iter[0].lower(),text.lower()) or search_nick(soft_iter[1],text.lower()):
			all_soft.append(soft_iter[0])
	for group_iter in group:
		if string_found(group_iter[0].lower(),text.lower())or search_nick(group_iter[1],text.lower()):
			all_group.append(group_iter[0])
	print(all_group,all_soft)
	return all_group,all_soft

group_software_parser("apt29 is a babyshark gkgkgk Blackfly")
