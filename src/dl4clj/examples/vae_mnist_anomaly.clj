(ns ^{:doc "recreated from: https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/anomalydetection/VaeMNISTAnomaly.java"}
    vae-mnist-anomaly
  (:require [dl4clj.nn.conf.builders.multi-layer-builders :as mlb]
            [dl4clj.nn.conf.builders.nn-conf-builder :as nn-conf]
            [dl4clj.nn.conf.builders.builders :as l]
            [dl4clj.examples.example-utils :as u])
  (:import [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           [org.deeplearning4j.berkeley Pair]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.dataset DataSet]
           [org.deeplearning4j.nn.conf.layers.variational VariationalAutoencoder$Builder]))

(def mini-batch-size 128)
(def rng-seed 12345)
(def n-epochs 5)
(def reconstruction-n-samples 16)

(def train-iter (MnistDataSetIterator. mini-batch-size true rng-seed))
;; not sure where this get used but it works
(.setSeed (Nd4j/getRandom) rng-seed)

;; cant continue without the @AllArgsConstructor annotation in the VariationalAutoencoder java file

(def conf
  (->
   (.build
    (nn-conf/nn-conf-builder {:seed rng-seed
                              :learning-rate 0.05
                              :updater :adam
                              :adam-mean-decay 0.9
                              :adam-var-decay 0.999
                              :weight-init :xavier
                              :regularization true
                              :l2 1e-4
                              :layers {0 (l/auto-encoder-layer-builder)} }))))
