from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

from data.read_data import *
from data.featurize import *
import pdb

class Model_Trainer:
   def __init__(self, model_type, split_type, task_type, feature_type, target_type, train_df, test_df):
      self.model_type = model_type
      self.split_type = split_type
      self.task_type = task_type
      self.feature_type = feature_type
      self.target_type = target_type
      self.train_df = train_df
      self.test_df = test_df
      self.has_test_targets = True
   
   def get_model_inputs(self):
      # read in everything
      target_train = self.train_df[self.target_type]
      #self.train_ketone_types = self.train_df['Ketone_Type'].to_list()
      self.y_train = get_target_data(target_train, self.task_type, self.target_type)
      try: #for the prospective studies, we don't have ground truth yields for the test set
         target_test = self.test_df[self.target_type]
      except: #if the test set doesn't have target values (prospective!)
         #self.y_test = gen_dummy_targets(self.test_df, self.task_type)
         self.has_test_targets = False
      else:
         self.y_test = get_target_data(target_test, self.task_type, self.target_type)

      #self.test_ketone_types = self.test_df['Ketone_Type'].to_list()
      ketone_train, ketone_test = self.train_df['Ketone_Smiles'].to_list(), self.test_df['Ketone_Smiles'].to_list()
      enzyme_train, enzyme_test = self.train_df['Enzyme'].to_list(), self.test_df['Enzyme'].to_list()

      # generate desired features and concatenate
      radius, nBits = 2, 2048

      if self.feature_type == 'ohe':
         ketone_train, ketone_test = np.array(ketone_train).reshape(-1,1), np.array(ketone_test).reshape(-1,1)
         enzyme_train, enzyme_test = np.array(enzyme_train).reshape(-1,1), np.array(enzyme_test).reshape(-1,1)
         ketone_ohe, enzyme_ohe = one_hot_encode(ketone_train), one_hot_encode(enzyme_train)
         ketone_ohe_train, enzyme_ohe_train = ketone_ohe.transform(ketone_train).toarray(), \
                                                         enzyme_ohe.transform(enzyme_train).toarray()
         ketone_ohe_test, enzyme_ohe_test = ketone_ohe.transform(ketone_test).toarray(), enzyme_ohe.transform(enzyme_test).toarray()
         self.X_train = ketone_ohe_train
         self.X_test = ketone_ohe_test

      elif self.feature_type == 'fgp':
         fgp_train = get_fgp_data(ketone_train, radius, nBits)
         fgp_test = get_fgp_data(ketone_test, radius, nBits)
         enzyme_train, enzyme_test = np.array(enzyme_train).reshape(-1,1), np.array(enzyme_test).reshape(-1,1)
         enzyme_ohe = one_hot_encode(enzyme_train)
         self.X_train = fgp_train
         self.X_test = fgp_test
      
      elif self.feature_type == 'physchem':
         physchem_train = get_physchem_descriptors(ketone_train)
         physchem_test = get_physchem_descriptors(ketone_test)
         scaler = MinMaxScaler().fit(physchem_train)
         physchem_train = scaler.transform(physchem_train)
         physchem_test = scaler.transform(physchem_test)
         enzyme_train, enzyme_test = np.array(enzyme_train).reshape(-1,1), np.array(enzyme_test).reshape(-1,1)
         enzyme_ohe = one_hot_encode(enzyme_train)
         self.X_train = physchem_train
         self.X_test = physchem_test

      elif self.feature_type == 'dft':
         dft_train = get_reactive_site_dft_data(self.train_df)
         dft_test = get_reactive_site_dft_data(self.test_df)
         scaler = MinMaxScaler().fit(dft_train)
         dft_train = scaler.transform(dft_train)
         dft_test = scaler.transform(dft_test)
         enzyme_train, enzyme_test = np.array(enzyme_train).reshape(-1,1), np.array(enzyme_test).reshape(-1,1)
         enzyme_ohe = one_hot_encode(enzyme_train)
         self.X_train = dft_train
         self.X_test = dft_test
      
      elif self.feature_type == 'physchemdft':
         physchem_train = get_physchem_descriptors(ketone_train)
         physchem_test = get_physchem_descriptors(ketone_test)
         scaler1 = MinMaxScaler().fit(physchem_train)
         physchem_train = scaler1.transform(physchem_train)
         physchem_test = scaler1.transform(physchem_test)

         dft_train = get_reactive_site_dft_data(self.train_df)
         dft_test = get_reactive_site_dft_data(self.test_df)
         scaler2 = MinMaxScaler().fit(dft_train)
         dft_train = scaler2.transform(dft_train)
         dft_test = scaler2.transform(dft_test)
         enzyme_train, enzyme_test = np.array(enzyme_train).reshape(-1,1), np.array(enzyme_test).reshape(-1,1)
         enzyme_ohe = one_hot_encode(enzyme_train)
         
         self.X_train = np.hstack((physchem_train, dft_train))
         self.X_test = np.hstack((physchem_test, dft_test))
      else:
         self.desired_dft_features = self.feature_type
         dft_train = get_dft_data(self.train_df, self.desired_dft_features)
         dft_test = get_dft_data(self.test_df, self.desired_dft_features)
         scaler = MinMaxScaler().fit(dft_train)
         dft_train = scaler.transform(dft_train)
         dft_test = scaler.transform(dft_test)
         self.X_train = dft_train
         self.X_test = dft_test
   
   def initialize_model(self):
      self.model_seed = 42
      if self.model_type == 'rf' and self.task_type == 'reg': self.model = RandomForestRegressor(random_state=self.model_seed)
      elif self.model_type == 'rf' and (self.task_type == 'bin' or self.task_type == 'mul'): self.model = RandomForestClassifier(random_state=self.model_seed)
      elif self.model_type == 'xgb' and self.task_type == 'reg': self.model = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.model_seed)
      elif self.model_type == 'xgb' and self.task_type == 'bin': self.model = xgb.XGBClassifier(objective='binary:logistic', random_state=self.model_seed)
      elif self.model_type == 'xgb' and self.task_type == 'mul': self.model = xgb.XGBClassifier(objective = 'multi:softprob', random_state=self.model_seed)
      elif self.model_type == 'lin' and self.task_type == 'reg': self.model = LinearRegression()
   
   def train_model(self):
      if self.model_type == 'nn': self.model.fit()
      else: self.model.fit(self.X_train,self.y_train)
   
   def model_predict(self):
      self.y_probs = None
      if self.task_type == 'bin': self.y_probs = self.model.predict_proba(self.X_test) 
      self.y_pred = self.model.predict(self.X_test)
   
   def train_test_model(self):
      self.get_model_inputs()
      self.initialize_model()
      self.train_model()
      self.model_predict()
