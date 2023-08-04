import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import urllib
import re
import os
import io
from sklearn.metrics import roc_auc_score, classification_report
import biotite.structure.io.pdbx as pdbx

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC
from tensorflow.errors import InvalidArgumentError
from tensorflow.keras.initializers import Ones
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import tensorflow as tf


from tensorflow.keras.layers import (
    Conv1D,
    Input,
    Dense,
    GlobalMaxPooling1D,
    Embedding, 
)

COLORS = ["#2d3142","#4f5d75","#bfc0c0","#ffffff","#ef8354"] 

class Tools:
    """Collection of helper functions used in different parts of the project"""
    
    # load data for protein label dictionaries (antry - entry name conversions)
    entries = pd.read_csv('data/uniprot_entries.tsv', sep='\t')
    entries = entries.set_index("Entry Name")["Entry"].to_dict()
    entries = defaultdict(lambda :"X", entries)
    entry_names = pd.read_csv('data/uniprot_entries.tsv', sep='\t')
    entry_names = entry_names.set_index("Entry")["Entry Name"].to_dict()
      
    @staticmethod
    def create_aa_dict(amino_acids=["ST"]):
        """creates amino acid dictionary to assign an integer to every amino acid

        Args:
            amino_acids (list, optional): List of amino acids which should be encoded. 0 value is assigned to amino acids which are not specified.
            Defaults to ["ST"].

        Returns:
            Dictionary: dictionary of (amino acid, integer value) pairs
        """        
        aa_dict = defaultdict(lambda: 0)
        for idx, aas in enumerate(amino_acids):
            for aa in aas:
                aa_dict[aa] = idx + 1
        return aa_dict
    
    @staticmethod
    def get_binding_region(seq, model, flanking_size=0):
        """Returns the protein region with the maximal convolution value, which is predicted to bind to the arrestin

        Args:
            seq (str): Protein sequence (one-letter amino acid labels)
            model (CNNModel): The trained model
            flanking_size (int, optional): The number of flanking amino acids before and after the binding region to return. Defaults to 0.

        Returns:
            str: The predicted arrestin-binding region
        """
        convoluted = model.convolute_sequence(seq)
        max_idx = np.argmax(convoluted)
        start = max_idx - flanking_size
        if start < 0:
            start = 0
        stop = max_idx + model.kernel_size + flanking_size
        if stop > len(seq):
            stop = len(seq)
        
        return seq[start:stop]
        
    @staticmethod
    def get_features(sequences, model):  
        """Converts sequences to model input features, each amino acid is assigned an integer according to the model's aa_dict

        Args:
            sequences (pandas DataFrame or list): dataframe of the sequences with the sequences in the first column, or list of sequences. If dataframe is used, the index is retained (e.g. protein names)
            model (CNNModel): trained model

        Returns:
            pandas DataFrame: dataframe with the converted sequences as rows
        """
        sequences = pd.DataFrame(sequences)
        idx = sequences.index
        sequences = [list(seq.values[0]) for name, seq in sequences.iterrows()]
        sequences = pd.DataFrame(sequences, index=idx)
        X = sequences.applymap(lambda x: model.aa_dict[x])
        return X
    
    @staticmethod
    def shuffle_data(X, y):
        """Shuffles the training data. 

        Args:
            X (pandas DataFrame): training features
            y (pandas Series): training targets with the same indices as X

        Returns:
            pandas DataFrame, Series: returns shuffled training features and targets
        """
        y = y.sample(frac=1, replace=False)
        X = X.loc[y.index]
        
        return X, y
    
    @staticmethod
    def cross_validate(name, model, sequences, targets, cv_sequences=None, patience=50,
                       repeats=10, split=0.2, converge_threshold=0.8, verbose=False):
        """Cross-validates the model

        Args:
            name (str or int): The name of the crossvalidation
            model (CNNModel): the model to run crossvalidation on
            sequences (pandas DataFrame or Series): Dataframe or Series of the sequences with the sequences in the first column if DataFrame.
            targets (pandas Series or numpy array): The targets (0 or 1) for the training.
            cv_sequences (pandas Series or DataFrame, optional): A pandas Series or DataFrame containing sequences other than the sequences used in the training. E.g. if full sequence is predicted but the training is run on shorter regions.
                                                                    Index should be the same as the in the sequences Series. Defaults to None.
            patience (int, optional): How long should the model run without improvement. Defaults to 50.
            repeats (int, optional): How many times will be the crossvalidation repeated. Defaults to 10.
            split (float, optional): Fraction of the split. Defaults to 0.2.
            converge_threshold (float, optional): The training auc threshold in each run. The training will be rerun until the threshold is reached to ensure convergence in the model. Defaults to 0.8.
            verbose (bool, optional): Verbosity of the training and cv aucs in each repeat, True or False. Defaults to False.

        Returns:
            pandas DataFrame: DataFrame of the crossvalidation results, 
                            columns:    name - (the provided name)
                                        dataset - train or cv auc
                                        repeat - the index of the repeat
                                        roc auc - roc auc of the prediction 
        """
        train_auc = [name, "train"]
        cv_auc = [name, "cv"]
        features = Tools.get_features(sequences, model)
        if cv_sequences is not None:
            cv_features = Tools.get_features(cv_sequences, model)
            no_of_cv_samples = int(len(features) * split)
        
        
        for i in range(repeats):
            
            X, y = Tools.shuffle_data(features, targets)
            model.fit(X, y,
                    epochs=200,
                    verbose=False,
                    validation_split=split,
                    class_weight={0: 0.5, 1: 0.5},
                    early_stop=True,
                    patience=patience,
                    converge_threshold=converge_threshold)
            train_auc.append(model.train_history.auc[-1])
            if cv_sequences is None:
                cv_auc.append(model.train_history.val_auc[-1])
            else:
                idxs = y.index[-no_of_cv_samples:]
                cv = cv_features.loc[idxs]
                pred = model.predict_proba(cv)
                try:
                    cv_auc.append(roc_auc_score(y.loc[idxs], pred))
                except ValueError: # if only one class gets to the cv set, the roc auc cannot be calculated
                    cv_auc.append(np.nan)
            if verbose:
                print(train_auc[-1], cv_auc[-1])
        results = pd.DataFrame([train_auc, cv_auc]).melt(id_vars=[0, 1])
        results.columns = ["name", "dataset", "repeat", "roc auc"]
        return results
    
    @staticmethod
    def get_uniprot_sequences(name):
        """This function takes in a protein uniprot name (str) as an input and returns the sequence from uniprot.org.
        If the uniprot website returns an HTTPError, it returns a string ">sp\n" instead.
        
        Args:
            name (str): uniprot name of the protein (entry or entry name)

        Returns:
            str: amino acid sequence of the protein
        """
                
        try:
            fp = urllib.request.urlopen(f"https://www.uniprot.org/uniprot/{name}.fasta?include=yes")
            mybytes = fp.read()

            fasta = mybytes.decode("utf8")
            fp.close()
        except urllib.request.HTTPError as exception:
            return ">sp\n"
        return Tools.parse_fasta(fasta)
    
    @staticmethod
    def parse_fasta(fasta):
        """Takes the fasta sequence from uniprot of a protein and returns the sequences in a list. 
        In case of variants multiple sequences are returned

        Args:
            fasta (str): fasta representation of a protein as downloaded from uniprot.org

        Returns:
            list of strings: list of protein sequences including variants from uniprot if available
        """
        result = []
        sequences = fasta.split(">sp")
        for sequence in sequences[1:]:
            sequence = sequence.split('\n')
            sequence = ''.join(sequence[1:])
            result.append(sequence)
        return result
    
    @staticmethod
    def get_uniprot_data(name):
        """Takes the uniprot name of a protein and downloads the corresponding text entry from the uniprot.org

        Args:
            name (str): entry of entry name of protein

        Returns:
            str: the protein entry text
        """
        try:
            fp = urllib.request.urlopen(f"https://www.uniprot.org/uniprot/{name}.txt")
            mybytes = fp.read()

            text = mybytes.decode("utf8")
            fp.close()
        except urllib.request.HTTPError as exception:
            return ">sp\n"
        return text
    
    @staticmethod
    def pxpp_present(seq):
        """Checks if the PxPP motif is present in the protein sequence, where P can be any potentially phosphorylated serine or threonine amino acid,
        according to https://doi.org/10.1101/2022.10.10.511556

        Args:
            seq (str): amino acid sequence of a protein

        Returns:
            bool: True or False, whether the PxPP is present in the sequence
        """
        return bool(len(re.findall('[S|T].[S|T][S|T]', seq)))
    
    @staticmethod
    def get_long_code(seq):
        """checks the number of the "long codes" in the given sequence. Long code is defined as described in https://doi.org/10.1016/j.cell.2017.07.002

        Args:
            seq (str): amino acid sequence of a protein

        Returns:
            int: number of long codes present in the sequence
        """
        return len(re.findall('[S|T]..[S|T][^P][^P][S|T|E|D]', seq))
    
    @staticmethod
    def get_short_code(seq):
        """checks the number of the "short codes" in the given sequence. Long code is defined as described in https://doi.org/10.1016/j.cell.2017.07.002

        Args:
            seq (str): amino acid sequence of a protein

        Returns:
            int: number of short codes present in the sequence
        """
        return len(re.findall('[S|T].[S|T][^P][^P][S|T|E|D]', seq))
        
    @staticmethod
    def get_alphafold_data(entry_name):
        """downloads alphafold .cif file for the input protein and extracts sequence and model confidence array for each amino acid position

        Args:
            entry_name (str): uniprot entry name of the protein

        Returns:
            tuple of sequence and confidence array: the protein sequence from the alphafold website with the model confidence array. 
                            If the entry is not available on the server, "x" insted of the sequence and an array of [0] as the confidence is returned
        """
        try:
            if "_" in entry_name:
                entry = Tools.entries[entry_name]
                
            else:
                entry = entry_name
            
            path = f"data/human_alphafold_pdbs/AF-{entry}-F1-model_v4.cif"
            if os.path.exists(path):
                cif = pdbx.PDBxFile.read(path)
            else:
                connection = urllib.request.urlopen(f"https://alphafold.ebi.ac.uk/files/AF-{entry}-F1-model_v4.cif")
                databytes = connection.read()
                connection.close()
                cif_txt = databytes.decode("utf8")

                f = io.StringIO(cif_txt)
                cif = pdbx.PDBxFile.read(f)
            
            confidence = pd.DataFrame(cif.get_category("ma_qa_metric_local")).metric_value.astype(float).values
            sequence = cif.get_category("entity_poly")["pdbx_seq_one_letter_code"]

            return sequence, confidence
        
        except:
            print(f"No structure for {entry_name}")
            return "x", np.array([0])
    
    @staticmethod
    def get_masked_sequence(entry_name, threshold=70):
        """creates masked sequence of the input protein. The amino acids with alphafold model confidence values above 70 are replaced with "X"

        Args:
            entry_name (str): the uniprot entry name of the protein
            threshold (int, optional): Alhpafold model confidence value. Above this value the amino acids are replaced with "X". Defaults to 70.

        Returns:
            str: The masked protein sequence
        """
        
        sequence, scores = Tools.get_alphafold_data(entry_name)
        masked = [aa if scores[idx] <70 else "X" for idx, aa in enumerate(sequence)]
        return "".join(masked)

class HistoryCallback(Callback):
    
    def __init__(self, model):
        """    

        Args:
            model (Tensorlow model): model
        """
        self.model = model

    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.auc = []
        self.val_auc = []

    def on_epoch_end(self, epoch, logs={}):
               
        if "acc" in logs.keys():
            self.acc.append(logs["acc"])
        if "val_acc" in logs.keys():
            self.val_acc.append(logs["val_acc"])
        if any(["auc" in x for x in logs.keys()]):
            key = [x for x in logs.keys() if "auc" in x][0]
            self.auc.append(logs[key])
        if any(["val_auc" in x for x in logs.keys()]):
            key = [x for x in logs.keys() if "val_auc" in x][0]
            self.val_auc.append(logs[key])

class CNNModel:
    
    def __init__(
        self,
        amino_acids=["ST"],
        kernel_size=12,
        learning_rate = 0.05,
        ):
        """the convolutional model to predict the arrestin-binding class of the receptors

        Args:
            amino_acids (list, optional): List of amino acids or amino acid groups which should be embedded. Groups can be defined as a single sequence containing the amino acid one-letter
                                        codes which should be embedded with the same values. Amino acids not defined in any groups will be assigned into the same group.
                                        Defaults to ["ST"] which results in two groups of amino acids, the S/T group and the rest.
            kernel_size (int, optional): The size of the kernel in the convulutional layer of the model. Defaults to 12.
            learning_rate (float, optional): Learning rate. Defaults to 0.05.
        """
        
        self.amino_acids = amino_acids
        self.kernel_size = kernel_size
        self.aa_dict = Tools.create_aa_dict(amino_acids=amino_acids)
        self.learning_rate = learning_rate
        
        self.classes_ = [0,1]
        #build the model
        self.build_model()
       
    def build_model(self):
        """method to build the model structure
        """
        init = Ones()
        
        seq = Input(shape=(None,), name="Input_Layer")
        
        embedding = Embedding(len(self.amino_acids)+1, 1, embeddings_regularizer=regularizers.L1(1e-4), name="Embedding_Layer")(seq)

        # we force the weigths with kernel_constraint to be positive for better interpretability
        conv_layer = Conv1D(1, self.kernel_size, strides=1, kernel_initializer=init, padding='valid', name="Conv1D_Layer")(embedding)
               
        globalmax = GlobalMaxPooling1D(name="GlobalMaxPooling1D_Layer")(conv_layer)

        prediction = Dense(1, activation="sigmoid", name="Prediction_Layer")(globalmax)
        
       
        model = Model(inputs=seq, outputs=prediction)
        globalmax_model = Model(seq, globalmax)
        embedding_model = Model(seq, embedding)
        convolution_model = Model(seq, conv_layer)
        
        sgd = SGD(learning_rate=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        
        model.compile(
            loss="binary_crossentropy",
            optimizer=sgd,
            metrics=["acc", AUC(name="auc")],
        )

        self.model, self.globalmax_model, self.embedding_model, self.convolution_model =  model, globalmax_model, embedding_model, convolution_model
        
    
    def fit(
        self,
        X,
        y,
        epochs=500,
        validation_split=0.0,
        verbose=False,
        class_weight={0: 1, 1: 1.0},
        converge_threshold=0.8,
        early_stop=False,
        patience=10
        ):
        """fits the model

        Args:
            X (pandas dataframe): pandas dataframe containing the training features
            y (pandas series): pandas series containing the targets
            epochs (int, optional): number of epochs. Defaults to 500.
            validation_split (float, optional): validation split. Defaults to 0.0.
            verbose (bool, optional): verbosity. Defaults to False.
            class_weight (dict, optional): weights of the two classes. Defaults to {0: 1, 1: 1.0}.
            converge_threshold (float, optional): roc auc value below which the training is not considered converged. Training will be repeated until over-threshold roc auc is achieved on the training set. Defaults to 0.8.
            early_stop (bool, optional): wheter to use early stop during the training. Defaults to False.
            patience (int, optional): number of the epochs of the patience in the early stop. Defaults to 10.
        """
        
        
        converged = False            
            
        # repeat the training until we get some convergence (e.g. the training auc is over 0.8)
        while not converged:

            self.build_model()

            self.train_history = HistoryCallback(self.model)
            callbacks = [self.train_history]
            
            # in case of an early stop, add it to the callbacks
            if early_stop:
                early = EarlyStopping(monitor='acc', 
                                    min_delta=0.02, patience=patience,
                                    mode='max',
                                    restore_best_weights=False)
                
                callbacks.append(early)

            train = self.model.fit(
                    X,
                    y,
                    epochs=epochs,
                    callbacks=callbacks,
                    use_multiprocessing=True,
                    validation_split=validation_split,
                    workers=16,
                    verbose=verbose,
                    class_weight=class_weight,

                )

            train_auc = self.train_history.auc[-1]

            if train_auc > converge_threshold:
                # print(f"converged {train_acc}")
                converged = True
            else:
                pass #print(train_auc)
            
        
    
    def convolute_sequence(self, sequence):
        """takes a protein sequence and runs a convolution on it with the trained weights

        Args:
            sequence (str): protein amino acid sequence

        Returns:
            array: convoluted sequence
        """
        
        sequence = np.array([self.aa_dict[aa] for aa in sequence]).reshape(1,-1)
        return self.convolution_model.predict([sequence]).reshape(-1)
    
    def predict(self, X):
        """predicts integer-converted sequences

        Args:
            X (pandas DataFrame): integer-converted sequences
            
        Returns:
            (int) prediction
        """        
        pred = self.model.predict(X)
        return (pred.reshape(-1)>0.5).astype(int)
    
    def predict_proba(self, X):
        """predicts integer-converted sequences

        Args:
            X (pandas DataFrame): integer-converted sequences
            
        Returns:
            (float): prediction as float between 0 and 1
        """        
        pred = self.model.predict(X)
        return pred.reshape(-1)
        
    def get_params(self, deep = False):
        """Helper method to make the model compatible with some sklearn utilities

        """
        return {'amino_acids': self.amino_acids, 
                'kernel_size': self.kernel_size,
                            }
    
    def display_training(self):
        """returns the plot of the training roc aucs
        """
        plt.plot(self.train_history.auc, c="black", label="train")
        plt.plot(self.train_history.val_auc, c="gray", label="cv")
        plt.legend()
        plt.ylim(0,1.1)
        sns.despine()
    
    def get_kernel(self):    
        """Returns the convolutional layer kernel weights"""
        return self.model.layers[2].get_weights()[0].reshape(-1)
    
    def display_kernel(self):
        """plots the convolutional layer kernel weights"""
        kernel = self.get_kernel()
        kernel = pd.DataFrame(pd.Series(kernel)).reset_index()
        kernel.columns = ['position', 'weight']
        sns.barplot(data=kernel, x = 'position', y='weight', color='#1A759F', edgecolor='black')
        sns.despine()
  
        
    def get_embeddings(self):
        """returns the embedding values of the amino acids"""
        aas = list("QWERTIPASDFGHKLYCVNM")
        sequence = "".join(aas)
        sequence_df = pd.DataFrame([["amino acids" , sequence]]).set_index(0)
        
        features = Tools.get_features(sequence_df, self)
        
        embedding = self.embedding_model.predict(features)
        
        embedding = pd.DataFrame(embedding.reshape(-1,1), index=list(sequence))
        embedding.reset_index(inplace=True)
        embedding.columns=["Amino acid", "Embedding value"]
        
        return embedding
    
    def display_embeddings(self):
        """displays the embedding values of the different amino acids"""
        embedding = self.get_embeddings()
        # plt.figure(figsize=(len(embedding)/2, 5))
        sns.barplot(data=embedding, x="Amino acid", y="Embedding value", color="#1A759F", edgecolor="black")
        sns.despine()

        
    def load_model(self, name):
        """loads a saved trained model from $PROJECT/models/dumps/ folder

        Args:
            name (str): the name of the model
        """        

        self.model.load_weights(f"models/dumps/{name}/model.model")
        self.globalmax_model.load_weights(f"models/dumps/{name}/globalmax_model.model")
        self.embedding_model.load_weights(f"models/dumps/{name}/embedding_model.model")
        self.convolution_model.load_weights(f"models/dumps/{name}/convolution_model.model")

        
    def save_model(self, name):
        """saves the model to $PROJECT/models/dumps/ folder

        Args:
            name (str): the name of the model
        """        
        try:
            os.mkdir(f"models/dumps/{name}")
        except:
            print("no new folder created, possibly it existed already?")
            
        self.model.save_weights(f"models/dumps/{name}/model.model")
        self.globalmax_model.save_weights(f"models/dumps/{name}/globalmax_model.model")
        self.embedding_model.save_weights(f"models/dumps/{name}/embedding_model.model")    
        self.convolution_model.save_weights(f"models/dumps/{name}/convolution_model.model")

