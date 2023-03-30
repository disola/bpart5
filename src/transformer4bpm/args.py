from transformer4bpm import syntax

class Args():
    def __init__(self):
        self.additional_special_tokens =["<"+str(type)+">" for type in syntax.NODE_CATEGORIES+["pool","lane"]]

        # PREPROCESSING
        self.model_filter_options = {
            'model_id': None, # we used: None
            'organization_id':None, # we used: None
            'activity': None, # we used: None
            'event': None, # we used: None
            'gateway': None, # we used: None
            'misc': None, # we used: None
            'model_name': None, # we used: None
            'label_length': (3,512), # we used: (3,512)
            'is_example_model': False, # we used: False
            'language': ['en'], # we used: ['en']
            'activity_event': None, # we used: None
            'node': (3,30) # we used: (3,30)
        }
        # use self.preprocess = False if you want to reproduce our experiments with the available files
        self.preprocess = False
        
        if self.preprocess: # change here, if you want to do your own preprocessing and generate new files
            self.USE_EXACT_SEQUENCE_LENGTH = True
            self.splitting = "model_id" # model_id or organization_id 
            self.SEQUENCE_LENGTH = 3 # sequence length without node to be predicted
            self.min_seq_eval = self.SEQUENCE_LENGTH  # wo node to be predicted 
            self.create_model_meta_data = True
            self.parse_bpmn_elements_for_networkx = True
            self.create_nx_graphs = True
            self.create_sequences = True
            self.create_simulations = True
            self.create_recommendation_cases = True
            self.do_splitting = "new_split" #"generate_sequences_from_split", "load_sequences_from_jsons", "new_split"   
        else: # do not change here, if you want to reproduce our experiments from the available files
            self.USE_EXACT_SEQUENCE_LENGTH = True
            self.splitting = "model_id" # model_id or organization_id 
            self.SEQUENCE_LENGTH = 3 # sequence length without node to be predicted
            self.min_seq_eval = self.SEQUENCE_LENGTH  # (wo node to be predicted -> for same as exact sequence lenght choose this length)
            self.create_model_meta_data = False
            self.parse_bpmn_elements_for_networkx = False
            self.create_nx_graphs = False
            self.create_sequences = False
            self.create_simulations = False
            self.create_recommendation_cases = False
            self.do_splitting = "load_sequences_from_jsons" #"generate_sequences_from_split", "load_sequences_from_jsons", "new_split"   

        # TRAINING
        self.model_params = {
            "model": "t5-small",
            "max_source_length": 512, # we used: 512
            "max_target_length": 128, # we used: 128
            "train_batch_size": 128,  # we used: 128
            "valid_batch_size": 128,  # we used: 128
            "train_epochs": 72,  # we used: 72
            "val_epochs": 72,  # we used: 72
            "learning_rate": 3e-4,  # we used: 3e-4
            "padding_side" : "left" # we used: left
            }

        # APPLICATION
        self.prediction_list_length = 10

