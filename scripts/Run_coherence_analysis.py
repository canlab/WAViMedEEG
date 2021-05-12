import sys
sys.path.append('..')
from src import Prep
from src import Standard
from src import Networks
from src import config
import os
from tqdm import tqdm
import argParser


def main():

    args = argParser.main([
        'studies_folder',
        'study_names',
        'reference_studies',
        'task',
        'filter_type',
        'frequency_band'
    ])

    studies_folder = args.studies_folder
    study_names = args.study_names
    reference_studies = args.reference_studies
    task = args.task
    filter_type = args.filter_type
    frequency_band = args.frequency_band

    # if reference_study is None:
    #     ref_studies = os.listdir(studies_folder)
    # else:
    ref_studies = [study+"/coherences/"+task for study in reference_studies]

    # if study_name is None:
    #     my_studies = os.listdir(studies_folder)
    # else:
    my_studies = [study+"/coherences/"+task for study in study_names]

    ref_coh = Networks.Coherence()
    for study in ref_studies:
        ref_coh.load_data(studies_folder+"/"+study)
    ref_coh.score(band=frequency_band)
    # ref_coh.draw(weighting=True)

    study_coh = Networks.Coherence()
    for study in my_studies:
        study_coh.load_data(studies_folder+"/"+study)
    study_coh.score(band=frequency_band)
    study_coh.threshold(reference=ref_coh.G, z_score=1)
    study_coh.draw(weighting=True, threshold=True)

if __name__ == '__main__':
    main()
