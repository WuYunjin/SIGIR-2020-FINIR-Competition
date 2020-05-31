import logging

import tensorflow as tf
import numpy as np
import pickle
import time
import os
from .issm import CompositeISSM
from ..utils import make_nd_diag,weighted_average
from tensorflow import Tensor
from sklearn.metrics import accuracy_score
from .lds import LDSArgsProj ,LDS
from .scaler import MeanScaler,NOPScaler
from .data_loader import DataLoader,GroundTruthLoader
from tqdm import tqdm
from .forecast import SampleForecast
import json
import matplotlib.pyplot as plt
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.model.forecast import SampleForecast
from gluonts.transform import (Chain ,
                               AddObservedValuesIndicator,
                               AddTimeFeatures,
                               AddAgeFeature,
                               VstackFeatures,
                               SwapAxes,
                               CanonicalInstanceSplitter,
                               InstanceSplitter)
from gluonts.transform.sampler import TestSplitSampler
from gluonts.dataset.field_names import FieldName
from .scaler import MeanScaler , NOPScaler
from gluonts.lzl_shared_ssm.utils import (plot_train_epoch_loss
, plot_train_pred_NoSSMnum
, del_previous_model_params
, complete_batch
, create_dataset_if_not_exist
, add_time_mark_to_file, add_time_mark_to_dir, get_model_params_name)
from gluonts.lzl_shared_ssm.models.data_loader import TrainDataLoader_OnlyPast ,InferenceDataLoader_WithFuture, mergeIterOut , stackIterOut


class FeatureEmbedder(object):
    def __init__(self,cardinalities,embedding_dims):
        assert (
                len(cardinalities) > 0
        ), "Length of `cardinalities` list must be greater than zero"
        assert len(cardinalities) == len(
            embedding_dims
        ), "Length of `embedding_dims` and `embedding_dims` should match"
        assert all(
            [c > 0 for c in cardinalities]
        ), "Elements of `cardinalities` should be > 0"
        assert all(
            [d > 0 for d in embedding_dims]
        ), "Elements of `embedding_dims` should be > 0"

        self.__num_features = len(cardinalities)

        def create_embedding(i: int, c: int, d: int) :
            embedding = tf.get_variable(name=f"cat_{i}_embedding" , shape=[c,d])
            return embedding

        with tf.variable_scope('feature_embedder'):
            self.__embedders = [
                create_embedding(i, c, d)
                for i, (c, d) in enumerate(zip(cardinalities, embedding_dims))
            ]

    def build_forward(self, features):
        '''
        :param features: Categorical features with shape: dynamic (N,T,C) or static (N,C), where C is the
            number of categorical features.
        :return: concatenated_tensor: Tensor
            Concatenated tensor of embeddings whth shape: (N,T,C) or (N,C),
            where C is the sum of the embedding dimensions for each categorical
            feature, i.e. C = sum(self.config.embedding_dims).
        '''
        if self.__num_features > 1:
            # we slice the last dimension, giving an array of length self.__num_features with shape (N,T) or (N)
            cat_feature_slices = tf.split(
                features, num_or_size_splits=self.__num_features,axis=-1
            )
        else:
            # F.split will iterate over the second-to-last axis if the last axis is one
            cat_feature_slices = [features]

        return tf.concat(
            [
                tf.nn.embedding_lookup(embed ,tf.dtypes.cast(tf.squeeze(cat_feature_slice, axis=-1) ,dtype=tf.int32))
                for embed, cat_feature_slice in zip(
                    self.__embedders, cat_feature_slices
                )
            ],
            axis=-1,
        )

class DeepStateNetwork(object):
    def load_original_data(self):
        name_prefix = 'data_process/processed_data/{}_{}_{}_{}.pkl'
        # 导入 target 以及 environment 的数据
        ds_names = ','.join([self.config.target, self.config.environment])
        if self.config.slice == 'overlap':
            series = self.config.timestep - self.config.past_length - self.config.pred_length + 1
            print('每个数据集的序列数量为 ,', series)
        elif self.config.slice == 'nolap':
            series = self.config.timestep // (self.config.past_length + self.config.pred_length)
            print('每个数据集的序列数量为 ,', series)
        else:
            series = 1
            print('当前序列数量为', series, ',是单长序列的情形')
        target_path = {name: name_prefix.format(
            '%s_start(%s)_freq(%s)' % (name, self.config.start, self.config.freq),
            '%s_DsSeries_%d' % (self.config.slice, series),
            'train_%d' % self.config.past_length, 'pred_%d' % self.config.pred_length,
        ) for name in self.config.target.split(',')}


        create_dataset_if_not_exist(
            paths=target_path, start=self.config.start, past_length=self.config.past_length
            , pred_length=self.config.pred_length, slice=self.config.slice
            , timestep=self.config.timestep, freq=self.config.freq
        )
        self.target_data = []
        # 由于 target 应该都是 dim = 1 只是确定有多少个 SSM 而已
        for target_name in target_path:
            target = target_path[target_name]
            with open(target, 'rb') as fp:
                target_ds = pickle.load(fp)
                assert target_ds.metadata['dim'] == 1, 'target 序列的维度都应该为1'
                self.target_data.append(target_ds)

        self.ssm_num = len(self.target_data)
        print('导入原始数据成功~~~')

    def transform_data(self):
        # 首先需要把 target
        time_features = time_features_from_frequency_str(self.config.freq)
        self.time_dim = len(time_features) + 1 # 考虑多加了一个 age_features
        seasonal = CompositeISSM.seasonal_features(self.freq)
        self.seasonal_dim = len(seasonal)
        transformation = Chain([
            SwapAxes(
                input_fields=[FieldName.TARGET],
                axes=(0, 1),
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # Unnormalized seasonal features
            AddTimeFeatures(
                time_features=seasonal,
                pred_length=self.pred_length,
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field="seasonal_indicators",
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features,
                pred_length=self.pred_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=self.pred_length,
                log_scale=True,
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                             + (
                                 [FieldName.FEAT_DYNAMIC_REAL]
                                 if self.use_feat_dynamic_real
                                 else []
                             ),
            ),
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                train_sampler=TestSplitSampler(),
                past_length=self.config.past_length,
                future_length=self.config.pred_length,
                output_NTC=True,
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES,
                    "seasonal_indicators"
                ],
                pick_incomplete=False,
            )

        ])
        print('已设置时间特征~~')
        # 设置环境变量的 dataloader
        target_train_iters = [iter(TrainDataLoader_OnlyPast(
            dataset=self.target_data[i].train,
            transform=transformation,
            batch_size=self.config.batch_size,
            num_batches_per_epoch=self.config.num_batches_per_epoch,
        )) for i in range(len(self.target_data))]
        target_test_iters = [iter(InferenceDataLoader_WithFuture(
            dataset=self.target_data[i].test,
            transform=transformation,
            batch_size=self.config.batch_size,
        )) for i in range(len(self.target_data))]

        self.target_train_loader = stackIterOut(target_train_iters,
                                                fields=[FieldName.OBSERVED_VALUES, FieldName.TARGET],
                                                dim=0,
                                                include_future=False)
        self.target_test_loader = stackIterOut(target_test_iters,
                                               fields=[FieldName.OBSERVED_VALUES, FieldName.TARGET],
                                               dim=0,
                                               include_future=True)

    def __init__(self, config , sess):
        self.config = config
        self.sess = sess

        # dataset configuration
        self.freq = config.freq
        self.past_length = config.past_length
        self.pred_length = config.pred_length
        self.add_trend = config.add_trend

        # network configuration
        self.batch_size = config.batch_size
        self.num_layers = config.num_layers
        self.num_cells = config.num_cells
        self.cell_type = config.cell_type
        self.dropout_rate = config.dropout_rate

        #prediciton configuration
        self.num_sample_paths = config.num_eval_samples
        self.scaling = config.scaling
        self.use_feat_dynamic_real = config.use_feat_dynamic_real
        self.use_feat_static_cat = config.use_feat_static_cat
        self.cardinality = [int(num) for num in config.cardinality.split(',')] if self.use_feat_static_cat else [1]
        embed_split = config.embedding_dimension.split(',')
        self.embedding_dimension = [int(num) for num in embed_split] if len(embed_split) > 1 \
            else [min(50, (cat + 1) // 2) for cat in self.cardinality]

        self.issm = CompositeISSM.get_from_freq(self.freq, self.add_trend)

        self.univariate = self.issm.output_dim() == 1

        # 放入数据集
        self.load_original_data()
        self.transform_data()


        # 开始搭建有可能Input 对应的 placeholder
        self.feat_static_cat = tf.placeholder(dtype=tf.float32,shape=[self.batch_size , 1])
        self.past_observed_values = tf.placeholder(dtype=tf.float32 , shape=[self.batch_size ,self.past_length, 1])
        self.past_seasonal_indicators = tf.placeholder(dtype=tf.float32 , shape = [self.batch_size,self.past_length , self.seasonal_dim])
        self.past_time_feat = tf.placeholder(dtype=tf.float32 , shape =[self.batch_size ,self.past_length, self.time_dim])
        self.past_target = tf.placeholder(dtype=tf.float32 , shape =[self.batch_size ,self.past_length, 1])
        self.future_seasonal_indicators = tf.placeholder(dtype=tf.float32 , shape=[self.batch_size ,self.pred_length, self.seasonal_dim])
        self.future_time_feat = tf.placeholder(dtype=tf.float32 , shape=[self.batch_size ,self.pred_length, self.time_dim])

    def build_module(self):
        with tf.variable_scope('deepstate', initializer=tf.contrib.layers.xavier_initializer() , reuse=tf.AUTO_REUSE):
            self.prior_mean_model = tf.layers.Dense(units=self.issm.latent_dim(),dtype=tf.float32 ,name='prior_mean')

            self.prior_cov_diag_model = tf.layers.Dense(
                units=self.issm.latent_dim(),
                dtype=tf.float32,
                activation=tf.keras.activations.sigmoid,
                name='prior_cov'
            )

            self.lds_proj = LDSArgsProj(output_dim=self.issm.output_dim())

            self.lstm = []
            for k in range(self.num_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_cells)
                cell = tf.nn.rnn_cell.ResidualWrapper(cell) if k > 0 else cell
                cell = (
                    tf.nn.rnn_cell.DropoutWrapper(cell,state_keep_prob=1.0-self.dropout_rate)
                    if self.dropout_rate > 0.0
                    else cell
                )
                self.lstm.append(cell)

            self.lstm = tf.nn.rnn_cell.MultiRNNCell(self.lstm)

            self.embedder = FeatureEmbedder(
                # cardinalities=cardinality,
                cardinalities=self.cardinality,
                embedding_dims=self.embedding_dimension,
            )

            # print(self.embedder.)
            # exit()

            if self.scaling:
                self.scaler = MeanScaler(keepdims=False)
            else:
                self.scaler = NOPScaler(keepdims=False)

            return self

    # 将在 build_forward 里面进行调用
    def compute_lds(
        self,
        feat_static_cat: Tensor,
        seasonal_indicators: Tensor,
        time_feat: Tensor,
        length: int,
        prior_mean = None, # can be None
        prior_cov = None, # can be None
        lstm_begin_state = None, # can be None
    ):
        # embed categorical features and expand along time axis

        embedded_cat = self.embedder.build_forward(feat_static_cat)

        repeated_static_features = tf.tile(
            tf.expand_dims(embedded_cat, axis=1),
            multiples=[1,length,1]
        ) #(bs , seq_length , embedding)


        # construct big features tensor (context)
        features = tf.concat([time_feat, repeated_static_features], axis=2)#(bs,seq_length , time_feat + embedding)

        # output 相当于使用 lstm 产生 SSM 的参数, 且只是一部分的参数
        output , lstm_final_state = tf.nn.dynamic_rnn(
            cell = self.lstm,
            inputs = features,
            initial_state = lstm_begin_state,
            dtype = tf.float32
        ) #(bs,seq_length ,hidden_dim)


        if prior_mean is None:
            prior_input = tf.squeeze(
                tf.slice(output, begin=[0, 0, 0], size=[-1, 1, -1]),
                axis=1
            ) #选用lstm 第一个时间步的结果

            prior_mean = self.prior_mean_model(prior_input)#(bs,latent_dim)
            prior_cov_diag = self.prior_cov_diag_model(prior_input)
            prior_cov = make_nd_diag( prior_cov_diag, self.issm.latent_dim())#(bs ,latent_dim ,latent_dim)


        emission_coeff, transition_coeff, innovation_coeff = self.issm.get_issm_coeff(
            seasonal_indicators
        )

        # emission_coeff（bs , seq_length , output_dim , latent_dim）
        # transition_coeff(bs, seq_length, latent_dim ,latent_dim)
        # innovation_coeff(bs ,seq_length ,latent_dim)


        noise_std, innovation, residuals = self.lds_proj.build_forward(output)
        # noise_std（bs , seq_length , 1 ）
        # innovation(bs, seq_length,  1 )
        # residuals(bs ,seq_length ,output_dim)



        lds = LDS(
            emission_coeff=emission_coeff,
            transition_coeff=transition_coeff,
            innovation_coeff=tf.math.multiply(innovation, innovation_coeff),
            noise_std=noise_std,
            residuals=residuals,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            latent_dim=self.issm.latent_dim(),
            output_dim=self.issm.output_dim(),
            seq_length=length,
        )

        return lds, lstm_final_state


    def build_train_forward(self):
        lds, _ = self.compute_lds(
            feat_static_cat=self.feat_static_cat,
            seasonal_indicators=tf.slice(
                self.past_seasonal_indicators
                ,begin=[0,0,0],size=[-1,-1,-1]
            ),
            time_feat=tf.slice(self.past_time_feat
                , begin=[0, 0, 0], size=[-1, -1, -1]
            ),
            length=self.past_length,
        )

        _, scale = self.scaler.build_forward(data=self.past_target
                               , observed_indicator=self.past_observed_values)
        #(bs,output_dim)


        observed_context = tf.slice(
            self.past_observed_values,
             begin=[0,0,0], size=[-1,-1,-1]
        )#(bs,seq_length , 1)


        ll, _, _ , mu_pred ,_ = lds.log_prob(
            x=tf.slice(
                self.past_target,
                begin=[0,0,0], size=[-1,-1,-1]
            ),#(bs,seq_length ,time_feat)
            observed=tf.math.reduce_min(observed_context ,axis=-1), #(bs ,seq_length)
            scale=scale,
        )#(bs,seq_length), (bs, seq , dim_l)
        mu_pred.set_shape([self.batch_size , self.config.past_length ,self.issm.latent_dim()])

        # 对于观测值的预测
        self.z_pred = tf.squeeze(tf.linalg.matmul(
            tf.transpose(lds.emission_coeff , [1,0,2,3]) #(bs, seq , 1 , dim_l)
            , tf.expand_dims(mu_pred, -1) #(bs ,seq ,dim_l ,1)
        ),axis=-1)#(bs ,seq ,1)
        self.z_pred = tf.math.multiply(
            self.z_pred,
            tf.expand_dims(scale, axis=1)
        )

        self.train_result = weighted_average(
            x=-ll, axis=1, weights= tf.math.reduce_min(observed_context, axis=-1)
        )#(bs,)

        self.train_result_mean = tf.math.reduce_mean(self.train_result)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        gvs = optimizer.compute_gradients(self.train_result_mean)
        capped_gvs = [(tf.clip_by_value(grad, -1., 10.), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)
        # tf.squeeze(observed_context, axis=-1)

        return self

    def build_predict_forward(self):
        lds, lstm_state = self.compute_lds(
            feat_static_cat=self.feat_static_cat,
            seasonal_indicators=tf.slice(
                self.past_seasonal_indicators
                ,begin=[0,0,0],size=[-1,-1,-1]
            ),
            time_feat=tf.slice(self.past_time_feat
                , begin=[0, 0, 0], size=[-1, -1, -1]
            ),
            length=self.past_length,
        )



        _, scale = self.scaler.build_forward(self.past_target, self.past_observed_values)

        observed_context = tf.slice(
            self.past_observed_values,
             begin=[0,0,0], size=[-1,-1,-1]
        )#(bs,seq_length , 1)

        _, final_mean, final_cov,_,_ = lds.log_prob(
            x=tf.slice(
                self.past_target,
                begin=[0, 0, 0], size=[-1, -1, -1]
            ),  # (bs,seq_length ,time_feat)
            observed=tf.math.reduce_min(observed_context, axis=-1),  # (bs ,seq_length)
            scale=scale,
        )

        # print('lstm final state shape : ',
        #       [{'c_shape': state.c.shape, 'h_shape': state.h.shape} for state in lstm_state])

        lds_prediction, _ = self.compute_lds(
            feat_static_cat=self.feat_static_cat,
            seasonal_indicators=self.future_seasonal_indicators,
            time_feat=self.future_time_feat,
            length=self.pred_length,
            lstm_begin_state=lstm_state,
            prior_mean=final_mean,
            prior_cov=final_cov,
        )

        samples = lds_prediction.sample(
            num_samples=self.num_sample_paths, scale=scale
        )# (num_samples, batch_size, seq_length, obs_dim)


        # (batch_size, num_samples, pred_length, target_dim)
        #  squeeze last axis in the univariate case (batch_size, num_samples, pred_length)
        if self.univariate:
            self.predict_result = tf.squeeze(tf.transpose(samples , [1, 0, 2, 3]),axis=3)
        else:
            self.predict_result = tf.transpose(samples , [1,0,2,3])


        return self


    def initialize_variables(self):
        """ Initialize variables or load saved models
        :return: self
        """
        # Setup saver
        self.saver = tf.train.Saver()

        # Initialize or reload variables
        if self.config.reload_model is not '':
            if self.config.reload_time == '':
                model_params = 'model_params'
            else:
                model_params = 'model_params_{}'.format(self.config.reload_time)
            params_path = os.path.join(self.config.logs_dir, self.config.reload_model, model_params)
            print("Restoring model in %s" % params_path)
            param_name = get_model_params_name(params_path)
            params_path = os.path.join(params_path, param_name)
            self.saver.restore(self.sess, params_path)
        else:
            print("Training to get new models params")
            self.sess.run(tf.global_variables_initializer())
        return self

    def train(self):
        #如果导入已经有的信息,不进行训练
        if self.config.reload_model != '':
            return
        sess = self.sess
        self.train_log_path = os.path.join(self.config.logs_dir,
                                   '_'.join([
                                            'freq(%s)' % (self.config.freq)
                                            ,'past(%d)'%(self.config.past_length)
                                            ,'pred(%d)'%(self.config.pred_length)
                                            , 'LSTM(%d-%d)' % (
                                                self.config.num_layers, self.config.num_cells)
                                            ,'trend(%s)' %(str(self.config.add_trend))
                                            , 'epoch(%d)' % (self.config.epochs)
                                            , 'bs(%d)' % (self.config.batch_size)
                                            , 'bn(%d)' % (self.config.num_batches_per_epoch)
                                            , 'lr(%s)' % (str(self.config.learning_rate))
                                            , 'dropout(%s)' % (str(self.config.dropout_rate))
                                     ])
                                           )
        params_path = os.path.join(self.train_log_path, 'model_params')
        params_path = add_time_mark_to_dir(params_path)
        print('训练参数保存在 --->', params_path)
        if not os.path.isdir(self.train_log_path):
            os.makedirs(self.train_log_path)

        num_batches = self.config.num_batches_per_epoch
        best_epoch_info = {
            'epoch_no': -1,
            'metric_value': np.Inf
        }
        train_plot_points = {}
        for epoch_no in range(self.config.epochs):
            #执行一系列操作
            #产生新的数据集
            tic = time.time()
            epoch_loss = 0.0

            with tqdm(range(num_batches)) as it:
                for batch_no in it :
                    target_batch_input = next(self.target_train_loader)
                    for i in range(self.ssm_num):
                        feed_dict = {
                            self.past_seasonal_indicators: target_batch_input['past_%s' % ('seasonal_indicators')],
                            self.past_time_feat: target_batch_input['past_time_feat'],
                            self.feat_static_cat : i * np.ones(dtype=np.float32 , shape=(self.config.batch_size ,1)),
                            self.past_observed_values : target_batch_input['past_%s' % (FieldName.OBSERVED_VALUES)][i] ,
                            self.past_target : target_batch_input['past_target'][i]}

                        batch_output , _   = sess.run([self.train_result , self.train_op]
                                                , feed_dict=feed_dict)
                        # batch_pred  = sess.run(self.z_pred , feed_dict = feed_dict)
                        # plot_train_pred_NoSSMnum(path=self.train_log_path,data=target_batch_input['past_target'][i]
                        #                          ,pred=batch_pred,batch=batch_no,epoch=epoch_no,ssm_no =i
                        #                          ,plot_num=4,time_start=target_batch_input['start']
                        #                          , freq=self.freq)
                        epoch_loss += np.sum(batch_output)
                        avg_epoch_loss = epoch_loss/((batch_no+1)*self.config.batch_size*(i+1))

                    avg_epoch_loss = avg_epoch_loss
                    it.set_postfix(
                        ordered_dict={
                            "avg_epoch_loss":  avg_epoch_loss
                        },
                        refresh=False,
                    )
            toc = time.time()
            train_plot_points[epoch_no] = avg_epoch_loss
            logging.info(
                "Epoch[%d] Elapsed time %.3f seconds",
                epoch_no,
                (toc - tic),
            )

            logging.info(
                "Epoch[%d] Evaluation metric '%s'=%f",
                epoch_no,
                "epoch_loss",
                avg_epoch_loss,
            )
            logging.info(
                "Epoch[%d] the best of this training(before this epoch) '%d <==> %f'",
                epoch_no,
                best_epoch_info['epoch_no'],
                best_epoch_info['metric_value'],
            )

            if avg_epoch_loss < best_epoch_info['metric_value']:
                del_previous_model_params(params_path)
                best_epoch_info['metric_value'] = avg_epoch_loss
                best_epoch_info['epoch_no'] = epoch_no
                self.train_save_path = os.path.join(params_path
                                                    ,'best_{}_epoch({})_nll({})'.format(self.config.target.replace(',' ,'_'),
                                                         epoch_no,
                                                         best_epoch_info['metric_value'])
                                                    )
                self.saver.save(sess, self.train_save_path)
                #'/{}_{}_best.ckpt'.format(self.dataset,epoch_no)
        plot_train_epoch_loss(train_plot_points, self.train_log_path ,params_path.split('_')[-1])
        logging.info(
            f"Loading parameters from best epoch "
            f"({best_epoch_info['epoch_no']})"
        )

        logging.info(
            f"Final loss: {best_epoch_info['metric_value']} "
            f"(occurred at epoch {best_epoch_info['epoch_no']})"
        )

        # save net parameters
        logging.getLogger().info("End models training")

    def predict(self):
        sess = self.sess
        self.all_forecast_result = []
        # 如果是 训练之后接着的 predict 直接采用之前 train 产生的save_path
        if hasattr(self ,'train_save_path'):
            path = self.train_save_path
            try:
                self.saver.restore(sess, save_path=path)
                print('there is no problem in restoring models params')
            except:
                print('something bad appears')
            finally:
                print('whatever ! life is still fantastic !')
        eval_result_root_path = 'evaluate/results/{}_slice({})_past({})_pred({})'.format(self.config.target.replace(',' ,'_') , self.config.slice ,self.config.past_length , self.config.pred_length)
        model_result = [];

        if not os.path.exists(eval_result_root_path):
            os.makedirs(eval_result_root_path)
        model_result_path = os.path.join(eval_result_root_path, '{}.pkl'.format(
            'deepstate(%s)_' % (self.config.target)
            + (self.train_log_path.split('/')[-1] if hasattr(self, 'train_log_path') else self.config.reload_model)
        ))
        model_result_path = add_time_mark_to_file(model_result_path)
        print('当前结果保存在--->' , model_result_path)
        for batch_no, target_batch in enumerate(self.target_test_loader):
            print('当前做Inference的第{}个batch的内容'.format(batch_no))
            if target_batch != None:
                batch_concat = []
                for i in range(self.ssm_num):
                    target_batch_complete, bs = complete_batch(batch=target_batch, batch_size=self.config.batch_size)
                    feed_dict = {
                        self.past_seasonal_indicators: target_batch_complete['past_%s' % ('seasonal_indicators')],
                        self.past_time_feat: target_batch_complete['past_time_feat'],
                        self.feat_static_cat: i * np.ones(dtype=np.float32, shape=(self.config.batch_size, 1)),
                        self.past_observed_values: target_batch_complete['past_%s' % (FieldName.OBSERVED_VALUES)][i],
                        self.past_target: target_batch_complete['past_target'][i],
                        self.future_time_feat : target_batch_complete['future_time_feat'],
                        self.future_seasonal_indicators : target_batch_complete['future_%s' % ('seasonal_indicators')]
                    }

                    samples_predict = sess.run(self.predict_result,feed_dict =  feed_dict)[:bs]
                    samples_predict = np.expand_dims(samples_predict ,axis=0) #(1 ,bs ,samples, seq)
                    batch_concat.append(samples_predict)

                batch_concat = np.concatenate(batch_concat , axis=0) #(ssm_num ,bs , samples, seq)
                model_result.append(batch_concat)

        model_result = np.concatenate(model_result , axis=1)
        with open(model_result_path, 'wb') as fp:
            pickle.dump(model_result, fp)

        return self
