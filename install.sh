module load cuda
module load git-lfs
pip install torch~=2.3.0
pip install -r requirements.txt
python -m wn download oewn:2023
python -m spacy download en_core_web_sm

