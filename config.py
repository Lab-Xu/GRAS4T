
class Config:
    def __init__(self):
        # General settings
        self.use_gpu = True
        self.platform = '10X'
        self.dataset_name = 'DLPFC'
        self.slice = '151507'
        self.have_label = False
        self.num_cluster = 'None'
        
        # Data preprocessing settings
        self.use_image = True
        self.save_path = './Results'
        self.cnnType_ = 'ResNet50'
        self.n_top_genes = 3000
        
        # Adjacency matrix construction settings
        self.k_X_ = 10
        self.k_C_ = 10
        self.weight_ = 0.5
        
        # Data augmentation settings
        self.aug1 = 'mask'
        self.aug2 = 'HS_image'
        self.drop_percent1 = 0.1
        self.drop_percent2 = 0.1
        self.image_k = 100
        
        # Training parameters
        self.nb_epochs_ = 500
        self.lr_ = 0.001
        self.patience_ = 20
        self.interval_num = 100
        self.is_select_neighbor = False
        self.select_neighbor = 5
        self.weight_lg = 1.0
        self.weight_lc = 1.0
        self.weight_recon = 1.0
        
        # Clustering parameters
        self.cluster_method = 'mclust'