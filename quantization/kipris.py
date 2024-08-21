#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import chardet


# # 불러오기

# In[2]:


first = [str(x) for x in Path('./file21').glob('*')]
print(first)
second = [str(x) for x in Path(f'./{first[0]}/').glob('*')]
third = [str(x) for x in Path(f'./{second[0]}/').glob('*')]
# INDIVIDUAL FILES
fourth = [str(x) for x in Path(f'./{third[0]}/').glob('*')]
# XML
fifth = [str(x) for x in Path(f'./{fourth[1]}/').glob('*')]


# # 인코딩 탐지

# In[3]:


with open(fifth[0], 'rb') as file:
    det = chardet.detect(file.read())
    print(det)



# # 파싱

# In[4]:


with open(fifth[0], 'r', encoding='euc-kr') as file:
    tree = ET.parse(file, parser=ET.XMLParser(encoding='EUC-KR'))
    root = tree.getroot()


# In[5]:


import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict
# XML 파일 로드하기

def parsing_xml(root):
    data = []

    document_types = root.findall('.//KR_DocumentType')
    for doc_type in document_types:
        document_type = doc_type.text

        # KR_ApplicationInformation 태그 정보 추출
        application_info = root.find('.//KR_ApplicationInformation')
        application_number = application_info.find('.//KR_ApplicationNumber').text
        application_date = application_info.find('.//KR_ApplicationDate').text

        # KR_Inventor 태그 정보 추출 
        inventors = root.findall('.//KR_InventorInformation')
        inventor_names = [inventor.find('.//KR_Name').text for inventor in inventors]
        inventor_address = [inventor.find('.//KR_Address').text for inventor in inventors]
        inventor_postcode = [inventor.find('.//KR_Postcode').text for inventor in inventors]
        inventor_country = [inventor.find('.//KR_Country').text for inventor in inventors]

        # invention-title 태그 정보 추출
        invention_title = root.find('.//invention-title').text

        # technical-field 태그 정보 추출
        technical_field = root.find('.//technical-field').find('.//p').text

        # background-art 태그 정보 추출
        background_art_paragraphs = root.findall('.//background-art//p')
       
        background_art = [paragraph.text for paragraph in background_art_paragraphs] 

        # <tech-problem> 태그 정보 추출
        tech_problem_paragraphs = root.findall('.//tech-problem//p')
        tech_problem = [paragraph.text for paragraph in tech_problem_paragraphs]  

        # <KR_OpenNumber> 
        opennumber = root.find('.//KR_OpenNumber').text

        #KR_OpenDate
        opendate = root.find('.//KR_OpenDate').text

        # KR_Applicant 태그 정보 추출
        applicant = root.find('.//KR_ORGName').text if root.find('.//KR_ORGName') != None else ''
            
        # KR_OpenNumber 및 KR_OpenDate 태그 정보 추출
        publication_number = root.find('.//KR_OpenNumber').text
        publication_date = root.find('.//KR_OpenDate').text

        # KR_IPCInformation 태그의 분류 정보 추출 (복수의 IPC가 있을 수 있으므로 리스트 사용)
        ipc_classifications = root.findall('.//KR_IPCInformation//KR_MainClassification')
        ipc_codes = [ipc.attrib['value'] for ipc in ipc_classifications]

        # tech-solution 태그 정보 추출
        tech_solutions = root.findall('.//tech-solution//p')
        tech_solution = [x.text for x in tech_solutions]

        # advantageous_effects 태그 정보 추출
        advantageous_effects = root.findall('.//advantageous-effects//p')
        advantageous_effect = [x.text for x in advantageous_effects]

        # description-of-drawings태그 정보 추출
        description_of_drawings = root.findall('.//description-of-drawings//p')
        description_of_drawing = [x.text for x in description_of_drawings]

        # description-of-embodiments 태그 정보 추출
        description_of_embodiments = root.findall('.//description-of-embodiments//p')
        description_of_embodiment = [x.text for x in description_of_embodiments]

        # reference-signs-list 태그 정보 추출
        reference_signs_list = root.findall('.//reference-signs-list//p')
        reference_signs = [x.text for x in reference_signs_list]

        # claim-text 태그 정보 추출
        claims_text = root.findall('.//claim//claim-text')
        claims = [x.text for x in claims_text]

        # summary 태그 정보 추출
        summary_text = root.findall('.//summary//p')
        summary = [x.text for x in summary_text]

        # drawing 태그 정보 추출
        drawings_text = root.findall('.//drawings//figure//img')
        drawing = [x.text for x in drawings_text]

        data_row = {
            'KR_Address': inventor_address,
            'KR_Name': inventor_names,
            'KR_invention_title': invention_title,
            'Technical_field' : technical_field,
            'background_art' : background_art,
            'tech_problem' : tech_problem,
            'Technical Solution': tech_solution,
            'Advantageous Effects': advantageous_effect,
            'Description of Drawings': description_of_drawing,
            'Description of Embodiments': description_of_embodiment,
            'Reference Signs List': reference_signs,
            'Claims': claims,
            'Summary': summary,
            'Drawings': drawing,
            'KR_PostCode': inventor_postcode,
            'KR_Country': inventor_country,
            'KR_OpenDate' : opendate,
            'Document Type': document_type,
            'KR_OpenNumber': opennumber,
            'Application Number': application_number,
            'Application Date': application_date,
            'Inventor Name': inventor_names,
            'Applicant': applicant,
            'Publication Number': publication_number,
            'Publication Date': publication_date,
            'IPC Codes': ', '.join(ipc_codes)
        }
        data.append(data_row)

    # 빈 데이터 리스트를 DataFrame으로 변환
    df = pd.DataFrame(data)
    
    return data


# In[27]:


datas = []
from tqdm import tqdm
for i in tqdm(second):
    for j in [str(x) for x in Path(f'./{second[0]}/').glob('*')]:
        xml = [str(x) for x in Path(f'./{j}/XML').glob('*.xml')][0]
        with open(xml, 'rb') as file:
            det = chardet.detect(file.read())

        with open(xml, 'r', encoding=det['encoding']) as file:
            tree = ET.parse(file, parser=ET.XMLParser(encoding=det['encoding']))
            root = tree.getroot()
            datas.append(parsing_xml(root))


# In[28]:


dd = defaultdict(list)

for d in datas:
    for key, value in d[0].items():
        dd[key].append(value)


# In[29]:


pd.DataFrame(dd) 


# In[ ]:




