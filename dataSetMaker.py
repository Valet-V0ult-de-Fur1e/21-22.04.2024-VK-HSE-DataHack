import requests as req 
from bs4 import BeautifulSoup
import pandas as pd
import time
from collections import deque


class RiaParser:
    '''Культура'''
    routes = [
        {
            'category': 'Политика',
            'keyword': 'politics'
        },
        {
            'category': 'Экономика',
            'keyword': 'economy'
        },
        {
            'category': 'Общество',
            'keyword': 'society'
        },
        {
            'category': 'Кино',
            'keyword': 'organization_Gruppa_Kino'
        },
        {
            'category': 'Законы',
            'keyword': 'tag_zakon'
        },
        {
            'category': 'Телевидение',
            'keyword': 'category_televidenie'
        },
        {
            'category': 'Люди',
            'keyword': 'keyword_ljudi'
        },
        {
            'category': 'Бренды',
            'keyword': 'keyword_brendy'
        },
        {
            'category': 'Наука',
            'keyword': 'tag_gadzhety'
        },
        {
            'category': 'Гаджеты',
            'keyword': 'tag_gadzhety'
        },
        {
            'category': 'Соцсети',
            'keyword': 'tag_Socseti'
        },
        {
            'category': 'Технологии',
            'keyword': 'technology'
        },
        {
            'category': 'Опросы',
            'keyword': 'polls'
        },
        {
            'category': 'Транспорт',
            'keyword': 'tag_thematic_category_Transport'
        },
        {
            'category': 'Погода',
            'keyword': 'category_pogoda'
        },
        {
            'category': 'Рецепты',
            'keyword': 'category_retsepty'
        },
        {
            'category': 'Мода',
            'keyword': 'tag_moda_2'
        },
        {
            'category': 'Красота',
            'keyword': 'product_krasota'
        },
    ]

    df = pd.DataFrame({'category': [], 'text': []})
    f = open('out_parsed_data.csv', 'a', encoding='utf-8')
    f.write('category')
    f.write(';')
    f.write('data')
    f.write('\n')
    deq = deque()

    @classmethod
    def get_ref_refs(cls, n):
        ret = []
        
        return ret
    
    @classmethod
    def get_refs_page(cls, start_url, n):
        pass


    @classmethod
    def get_refs_news(cls, soup):
        ret = []
        # soup = BeautifulSoup(html, 'html.parser')
        elements = soup.find_all('a', attrs={'class': 'list-item__title color-font-hover-only'})
        for element in elements:
            ret.append(element['href'])
        return ret
    
    @classmethod
    def get_text(cls, url):
        if 1:
            res = req.get(url)
            soup = BeautifulSoup(res.text, 'html.parser')

            topics = soup.find_all('div', attrs={'class': 'article__block'})
            topics_text = []
            for topic in topics:
                topics_text.append(topic.text)
            return ' '.join(topics_text)
        return ''

    @classmethod
    def save_to_df(cls, category, data):
        cls.df.loc[len(cls.df)] = [category, data]

    @classmethod
    def save_to_file(cls, category, data):
        cls.f.write(category)
        cls.f.write(';')
        cls.f.write(data)
        cls.f.write('\n')
    
    @classmethod
    def save_deque_to_file(cls):
        cls.deq
        while cls.deq:
            cat, text = cls.deq.pop()
            cls.f.write(cat)
            cls.f.write(';')
            cls.f.write(text)
            cls.f.write('\n')
        cls.f.flush()


    @classmethod
    def get_dataset(cls, start_cat=0, end_cat=len(routes)):
        for route in cls.routes[start_cat:end_cat]:
            category = route['category']
            keyword = route['keyword']
            url = 'https://ria.ru/' + keyword
            res = req.get(url)
            soup = BeautifulSoup(res.text, 'html.parser')
            for ref in cls.get_refs_news(soup):
                # cls.save_to_file(category, cls.get_text(ref))
                cls.deq.append((category, cls.get_text(ref)))
                # cls.save_to_df(category, cls.get_text(ref))
            # cls.f.flush()
            
            url = 'https://ria.ru' + soup.find('div', attrs={'class': 'list-more color-btn-second-hover'})['data-url']
            res = req.get(url)
            soup = BeautifulSoup(res.text, 'html.parser')
            for ref in cls.get_refs_news(soup):
                # cls.save_to_df(category, cls.get_text(ref))
                cls.deq.append((category, cls.get_text(ref)))
            

            for i in range(4):
                url = 'https://ria.ru' + soup.find('div', attrs={'class': 'list-items-loaded'})['data-next-url']
                res = req.get(url)
                soup = BeautifulSoup(res.text, 'html.parser')
                for ref in cls.get_refs_news(soup):
                    # cls.save_to_df(category, cls.get_text(ref))
                    cls.deq.append((category, cls.get_text(ref)))

            
            # with open('out.html', 'w', encoding='utf-8') as f:
            #     f.write(res.text)
            cls.save_deque_to_file()
            print('end', category)

start = time.time()
deq = RiaParser.get_dataset()
end = time.time()
print(end - start)

