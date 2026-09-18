"""Lista simples de stopwords em português; normaliza removendo acentos para
compatibilidade com `strip_accents='unicode'` do `TfidfVectorizer`.
"""
import unicodedata

_RAW = [
    'a','à','ao','aos','aquela','aquelas','aquele','aqueles','aquilo','as','às','até','com','como',
    'da','das','de','dela','delas','dele','deles','demais','depois','do','dos','e','ela','elas','ele',
    'eles','em','entre','era','essa','essas','esse','esses','esta','está','estar','estão','estas','estive',
    'estamos','estão','este','estes','eu','foi','for','foram','has','havia','isso','já','la','lá','lhe',
    'lhes','lo','mais','mas','me','mesmo','meu','meus','minha','minhas','muito','na','nas','não','nem',
    'no','nos','nossa','nossas','nosso','nossos','num','numa','o','os','ou','para','pela','pelas','pelo',
    'pelos','por','qual','quando','que','quem','se','sem','ser','seu','seus','sua','suas','também','te',
    'tem','tendo','tenho','ter','teu','teus','ti','tido','tinha','tive','tivemos','tiveram','tu','um','uma',
    'você','vocês','vos','àquele','àquela','às','ao','aos'
]


def _normalize(s: str) -> str:
    n = unicodedata.normalize('NFKD', s)
    return ''.join(ch for ch in n if not unicodedata.combining(ch)).lower()


STOPWORDS_PT = list(dict.fromkeys(_normalize(s) for s in _RAW))
