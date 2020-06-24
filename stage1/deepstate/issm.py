# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports

# Third-party imports
from pandas.tseries.frequencies import to_offset
import os
from tensorflow import Tensor
from ..utils import (_make_block_diagonal,_broadcast_param)
import tensorflow as tf

from gluonts.time_feature import (
    TimeFeature,
    MinuteOfHour,
    HourOfDay,
    DayOfWeek,
    DayOfMonth,
    WeekOfYear,
    MonthOfYear,
)

class ISSM:
    r"""
    An abstract class for providing the basic structure of Innovation State Space Model (ISSM).

    The structure of ISSM is given by

        * dimension of the latent state
        * transition and emission coefficents of the transition models
        * emission coefficient of the observation models

    """

    def __init__(self):
        pass

    def latent_dim(self) :
        raise NotImplemented()

    def output_dim(self):
        raise NotImplemented()

    def emission_coeff(self, seasonal_indicators):
        raise NotImplemented()

    def transition_coeff(self, seasonal_indicators):
        raise NotImplemented()

    def innovation_coeff(self, seasonal_indicators):
        raise NotImplemented()

    def get_issm_coeff(
        self, seasonal_indicators
    ) :
        return (
            self.emission_coeff(seasonal_indicators),
            self.transition_coeff(seasonal_indicators),
            self.innovation_coeff(seasonal_indicators),
        )


class LevelISSM(ISSM):
    def latent_dim(self) :
        return 1

    def output_dim(self) :
        return 1

    def emission_coeff(
        self, seasonal_indicators  # (batch_size, time_length)
    ) :

        _emission_coeff = tf.ones(shape=(1, 1, 1, self.latent_dim()), name="emission_coeff")

        # get the right shape: (batch_size, seq_length, obs_dim, latent_dim)
        zeros = _broadcast_param(
            tf.zeros_like(
                tf.squeeze(
                    tf.slice(seasonal_indicators
                             ,begin=[0]*len(seasonal_indicators.shape)
                             ,size=[-1]*(len(seasonal_indicators.shape)-1)+[1])
                ,axis =-1)
            ),
            axes=[2, 3],
            sizes=[1, self.latent_dim()],
        )

        return tf.broadcast_to(_emission_coeff , shape=zeros.shape)
            # _emission_coeff.broadcast_like(zeros)

    def transition_coeff(
        self, seasonal_indicators  # (batch_size, time_length)
    ) :

        _transition_coeff = (
            tf.expand_dims(
                tf.expand_dims(
                    tf.eye(self.latent_dim()),axis=0), axis=0)
        )#(1,1,lat,lat)


        # get the right shape: (batch_size, seq_length, latent_dim, latent_dim)
        zeros = _broadcast_param(
            tf.zeros_like(
                tf.squeeze(
                    tf.slice(seasonal_indicators
                             , begin=[0]*len(seasonal_indicators.shape)
                             , size=[-1]*(len(seasonal_indicators.shape) - 1) + [1])
                    , axis=-1)
            ),
            axes=[2, 3],
            sizes=[self.latent_dim(), self.latent_dim()],
        )

        return tf.broadcast_to(_transition_coeff , shape=zeros.shape)

    # 代表 隐状态 之间传递的 方差
    def innovation_coeff(
        self, seasonal_indicators  # (batch_size, time_length)
    ) :
        return tf.squeeze(self.emission_coeff(seasonal_indicators),axis=2)


class LevelTrendISSM(LevelISSM):
    def latent_dim(self) :
        return 2

    def output_dim(self) :
        return 1

    def transition_coeff(
        self, seasonal_indicators  # (batch_size, time_length)
    ) :

        _transition_coeff = (
            tf.expand_dims(
                tf.expand_dims(
                    tf.constant([[1., 1.], [0., 1.]])
                    , axis=0),axis=0)
        )

        # get the right shape: (batch_size, seq_length, latent_dim, latent_dim)
        zeros = _broadcast_param(
            tf.zeros_like(
                tf.squeeze(
                    tf.slice(seasonal_indicators
                             , begin=[0] * len(seasonal_indicators.shape)
                             , size=[-1] * (len(seasonal_indicators.shape) - 1) + [1])
                    , axis=-1)
            ),
            axes=[2, 3],
            sizes=[self.latent_dim(), self.latent_dim()],
        )

        return tf.broadcast_to(_transition_coeff , shape=zeros.shape)
            # _transition_coeff.broadcast_like(zeros)


class SeasonalityISSM(LevelISSM):
    """
    Implements periodic seasonality which is entirely determined by the period `num_seasons`.
    """

    def __init__(self, num_seasons) :
        super(SeasonalityISSM, self).__init__()
        self.num_seasons = num_seasons

    def latent_dim(self):
        return self.num_seasons

    def output_dim(self) :
        return 1

    def emission_coeff(self, seasonal_indicators):# input(bath_size , length , 1)
        return tf.one_hot(
                tf.dtypes.cast(seasonal_indicators,dtype=tf.int32)
                , depth=self.latent_dim()
                , dtype=tf.float32
            )  #output(bath_size , length , 1 , latent_dim)

    def innovation_coeff(self, seasonal_indicators) :
        return tf.squeeze(
            tf.one_hot(
                tf.dtypes.cast(seasonal_indicators,dtype=tf.int32)
                , depth=self.latent_dim()
                , dtype=tf.float32
            )
            ,axis=2
        )



class CompositeISSM(ISSM):
    DEFAULT_ADD_TREND = True

    def __init__(
        self,
        seasonal_issms,
        add_trend = DEFAULT_ADD_TREND,
    ) :
        super(CompositeISSM, self).__init__()
        self.seasonal_issms = seasonal_issms
        self.nonseasonal_issm = (
            LevelISSM() if add_trend is False else LevelTrendISSM()
        )

    def latent_dim(self) :
        return (
            sum([issm.latent_dim() for issm in self.seasonal_issms])
            + self.nonseasonal_issm.latent_dim()
        )

    def output_dim(self):
        return self.nonseasonal_issm.output_dim()

    @classmethod
    def get_from_freq(cls, freq, add_trend = DEFAULT_ADD_TREND):
        offset = to_offset(freq)

        seasonal_issms = []

        if offset.name == "M":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=12)  # month-of-year seasonality
            ]
        elif offset.name == "W-SUN":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=53)  # week-of-year seasonality
            ]
        #在这里给 seasonality 添加多一个 关于月份的周期性
        elif offset.name == "D":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=7),  # day-of-week seasonality
                SeasonalityISSM(num_seasons=31) # day-of-month seasonality
            ]
        elif offset.name == "B":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=7)  # day-of-week seasonality
            ]
        elif offset.name == "H":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=24),  # hour-of-day seasonality
                SeasonalityISSM(num_seasons=7),  # day-of-week seasonality
            ]
        elif offset.name == "T":
            seasonal_issms = [
                SeasonalityISSM(num_seasons=60),  # minute-of-hour seasonality
                SeasonalityISSM(num_seasons=24),  # hour-of-day seasonality
            ]
        else:
            RuntimeError("Unsupported frequency {}".format(offset.name))

        return cls(seasonal_issms=seasonal_issms, add_trend=add_trend)

    @classmethod
    def seasonal_features(cls, freq):
        offset = to_offset(freq)
        if offset.name == "M":
            return [MonthOfYear(normalized=False)]
        elif offset.name == "W-SUN":
            return [WeekOfYear(normalized=False)]
        elif offset.name == "D":
            return [DayOfWeek(normalized=False) ,DayOfMonth(normalized=False)]
        elif offset.name == "B":  # TODO: check this case
            return [DayOfWeek(normalized=False)]
        elif offset.name == "H":
            return [HourOfDay(normalized=False), DayOfWeek(normalized=False)]
        elif offset.name == "T":
            return [
                MinuteOfHour(normalized=False),
                HourOfDay(normalized=False),
            ]
        else:
            RuntimeError("Unsupported frequency {}".format(offset.name))

        return []

    def get_issm_coeff(
        self, seasonal_indicators  # (batch_size, time_length)
    )  :
        emission_coeff_ls, transition_coeff_ls, innovation_coeff_ls = zip(
            self.nonseasonal_issm.get_issm_coeff(seasonal_indicators),
            *[
                issm.get_issm_coeff(
                    tf.slice(seasonal_indicators ,begin=[0]*(len(seasonal_indicators.shape)-1)+[ix]
                             ,size=[-1]*(len(seasonal_indicators.shape)-1)+[1]
                             )
                )
                for ix, issm in enumerate(self.seasonal_issms)
            ] )

        # stack emission and innovation coefficients
        emission_coeff = tf.concat([coeff for coeff in emission_coeff_ls], axis=-1)


        innovation_coeff = tf.concat([coeff for coeff in innovation_coeff_ls], axis=-1)

        # transition coefficient is block diagonal!
        transition_coeff = _make_block_diagonal(transition_coeff_ls)
        # print(emission_coeff, '\n', innovation_coeff, '\n', transition_coeff)
        return emission_coeff, transition_coeff, innovation_coeff



