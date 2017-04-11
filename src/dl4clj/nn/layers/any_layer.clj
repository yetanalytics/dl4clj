(ns dl4clj.nn.layers.any-layer
  (:import [org.deeplearning4j.nn.layers ActivationLayer
            BaseLayer BaseOutputLayer BasePretrainNetwork
            DropoutLayer FrozenLayer LossLayer OutputLayer]
           [org.deeplearning4j.nn.layers.convolution
            Convolution1DLayer ConvolutionLayer ZeroPaddingLayer])
  (:require [dl4clj.nn.api.layer :as l]
            [dl4clj.nn.conf.utils :as u]))
;; figure out the best implemntation for these development fns
