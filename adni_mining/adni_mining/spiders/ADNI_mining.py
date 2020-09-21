import scrapy
from scrapy.crawler import CrawlerRunner
from twisted.internet import reactor as rt, defer
from getpass import getpass
from ipdb import set_trace
from scrapy.utils.response import open_in_browser as oib

class ADNISpider(scrapy.Spider):
    name = 'adni'
    login_url = 'https://ida.loni.usc.edu/login.jsp?project=ADNI'
    adv_search_url = 'https://ida.loni.usc.edu/pages/access/search.jsp?advSearch'
    
    def login(self, res):
        print(f'parsing login {res.url}')
        csrf = res.xpath('//*[@name="csrf_token"]/@value').extract_first()
        
        self.email = input('adni email')
        self.password = getpass('adni password')
        
        yield scrapy.FormRequest.from_response(res, formdata={
            'userEmail': self.email,
            'userPassword': self.password,
        }, callback=self.adv_tab)
        
    def adv_tab(self, res):
        print(f'parsing main page {res.url}')
        self.img_url = str(res.body).split('">Image Collections')[0].split('"')[-1]
        set_trace()
        yield scrapy.Request(f'{self.img_url}#tab4', callback=self.parse_search_res)
    
    def parse_search_res(self, res):
        print(f'parsing adv_search page {res.url}')
        oib(res)
        yield scrapy.FormRequest.from_response(res, formdata={
            'tab': 'advResult',
            'subjectOption': 'true',
            'visitOption': 'true',
            'imageModalityOption': 'true',
            'imgType': '4',
            'advOrderBy1': 'SUBJECT_ID',
            'advOrderBy2': '',
            'project_checkBox': 'ADNI',
            'projectPhase_checkBox': 'ADNI_1',
            'projectPhase_checkBox': 'ADNI_2',
            'projectPhase_checkBox': 'ADNI_3',
            'projectPhase_checkBox': 'ADNI_4',
            'advSubjectId': '',
            'displayInResult': 'SUBJECT.SUBJECT_ID',
            'advAgeMenu': 'equals',
            'advAge_textBox': '',
            'displayInResult': 'SUBJECT.AGE',
            'advSex': 'OTHER',
            'displayInResult': 'SUBJECT.SUBJECT_SEX',
            'advWeightMenu': 'equals',
            'weight_textBox': '',
            'researchGroup_checkBox': '31',
            'advStudyDate': '',
            'advStudyDate': '',
            'archiveDate': '',
            'archiveDate': '',
            'visit_andOr_ADNI': 'OR',
            'imgDesc': 'FreeSurfer Cross-Sectional Processing brainmask',
            'imgId': '',
            'imgModality_checkBox': '1',
            'imgModality_andOr': 'OR',
            'imgProtocol_1_Field_Strength_Menu': 'equals',
            'imgProtocol_1_Field_Strength_textBox': '',
            'imgProtocol_1_Matrix_Z_Menu': 'equals',
            'imgProtocol_1_Matrix_Z_textBox': '',
            'imgProtocol_1_Slice_Thickness_Menu': 'equals',
        }, cookies={
            'PROJECT_SECTION': 'true',
            'PROJECT_SPECIFIC_SECTION': 'false',
            'SUBJECT_SECTION': 'true',
            'ASSESSMENT_SECTION': 'false',
            'IMG_TYPE_PRE_PROCESS_SECTION': 'false',
            'MODALITY_SECTION': 'true',
            'PROTOCOL_SECTION': 'false',
            'QUALITY_SECTION': 'false',
            'STATUS_SECTION': 'false',
            'STUDY_VISIT_SECTION': 'true',
            'ADV_QUERY': 'true',
            'SORT_COLUMN': '9',
            'IS_FORWARD_SORT': 'true',
            'PROCESSING_SECTION': 'false',
            'IMG_TYPE_POST_PROCESS_SECTION': 'true',
            'IMG_TYPE_ORIG_SECTION': 'false',
            '_ga': 'GA1.2.1943246582.1596378396',
            '_gid': 'GA1.2.1158153468.1600503883',
            '__utmc': '174947263',
            '__utmz': '174947263.1600666881.16.13.utmcsr',
            '__utma': '174947263.1943246582.1596378396.1600671498.1600675019.18',
            '__utmt': '1',
            '__utmb': '174947263.23.10.1600675019'
        }, formname='advancedQuery', clickdata={'id': 'advSearchQuery', 'nr': 1}, callback=self.parse_brain_lst)
    
    def parse_brain_lst(self, res):
        print(f'parsing brain list {res.url}')
        oib(res)
        
    def start_requests(self):
        yield scrapy.Request(self.login_url, callback=self.login)
            
runner = CrawlerRunner()