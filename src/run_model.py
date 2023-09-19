from logging import getLogger
from recbole.utils import init_logger, init_seed
#from recbole.trainer import Trainer
from recbole.model import general_recommender, sequential_recommender, knowledge_aware_recommender
#from sklearn.model_selection import KFold
#from recbole.custom_model_general import NewModel
from custom_trainer import NewTrainer
from custom_model_sequential import CustomLSTM

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.data.utils import get_dataloader

import csv   
from datetime import datetime

import matplotlib.pylab as plt

if __name__ == '__main__':
    models = [
    sequential_recommender.STAMP,
    #sequential_recommender.BERT4Rec,
    #sequential_recommender.NARM,
    #sequential_recommender.NPE,
    #sequential_recommender.SASRec,
    #sequential_recommender.TransRec
    #sequential_recommender.STAMP,
    #CustomLSTM
    ]

    for model in models:  #configSequentialNew #configListSessionBased #configSequentialModels
        config = Config(model=model, config_file_list=['/home/eduardo/masters/MovieLens-100k/src/recbole/config/configListSessionBased.yml'])  
           
        init_seed(config['seed'], config['reproducibility'])
        
        # logger initialization
        init_logger(config)
        logger = getLogger()
        logger.info('---------------------------------')
        logger.info('config')
        logger.info(config)

        # dataset filtering
        dataset = create_dataset(config)
        logger.info('---------------------------------')
        logger.info('dataset')
        logger.info(dataset)
        
        # dataset splitting
        train_dataset, valid_dataset, test_dataset = dataset.build()

        train_data = get_dataloader(config, "train")(
            config, train_dataset, None, shuffle=False
        )
        test_data = get_dataloader(config, "test")(
            config, test_dataset, None, shuffle=False
        )
        valid_data = get_dataloader(config, "valid")(
            config, valid_dataset, None, shuffle=False
        )
        print(train_data)

        # model loading and initialization
        model = model(config, train_data.dataset).to(config['device'])
        logger.info('---------------------------------')
        logger.info('model')
        logger.info(model)

        # trainer loading and initialization
        trainer = Trainer(config, model) #NewTrainer(config, model)

        # val_metric_values = []
        # def val_callback(epoch_idx, valid_score):
        #     val_metric_values.append(valid_score)

        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True) #config['show_progress'] callback_fn=val_callback

        # model evaluation
        test_result = trainer.evaluate(test_data)
        model_unique_id = datetime.now().strftime("%Y-%m-%d-%H:%M:%S_") + model.__class__.__name__
    
        print(trainer.train_loss_dict)

        logger.info('---------------------------------')
        logger.info('best valid result: {}'.format(best_valid_result))
        logger.info('test result: {}'.format(test_result))
        logger.info('model name: {}'.format(model_unique_id))
        logger.info('---------------------------------')

       # best_valid_result['MRR'] # testar se isso funciona
        fields=[model.__class__.__name__,config['dataset'],datetime.now(), best_valid_result,test_result]  #log location, model location and name
        with open(r'data.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
