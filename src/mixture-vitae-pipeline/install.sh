apt-get install python3-dev
pip3 install -r requirements.txt --break-system-packages
python3 -m spacy download  en_core_web_sm --break-system-packages
python3 -m spacy download  xx_ent_wiki_sm  --break-system-packages
