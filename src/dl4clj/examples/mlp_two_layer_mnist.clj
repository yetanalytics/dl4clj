(ns ^{:doc "recreation of: https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistTwoLayerExample.java"}
    mlp-two-layer-mnist
  (:require [dl4clj.nn.conf.builders.multi-layer-builders :as mlb]
            [dl4clj.nn.conf.builders.nn-conf-builder :as nn-conf]
            [dl4clj.nn.conf.builders.builders :as l]
            [dl4clj.examples.example-utils :as u])
  (:import [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           [org.deeplearning4j.eval Evaluation]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.dataset DataSet]
           [org.slf4j Logger]
           [org.slf4j LoggerFactory]
           ))

(def num-rows 28)
(def num-cols 28)
(def output-n 10)
(def batch-size 64)
(def rng-seed 123)
(def num-epochs 15)
(def l-rate 0.0015)

(def train-iterator (MnistDataSetIterator. batch-size true rng-seed))
(def test-iterator (MnistDataSetIterator. batch-size false rng-seed))

(def conf
  (->
   (.build
    (nn-conf/nn-conf-builder
     {:seed rng-seed
      :optimization-algo :stochastic-gradient-descent
      :iterations 1
      :global-activation-fn :relu
      :weight-init :xavier
      :learning-rate l-rate
      :updater :nesterovs
      :momentum 0.98
      :regularization true
      :l2 (* l-rate 0.005)
      :layers {0 (l/dense-layer-builder {:n-in (* num-rows num-cols)
                                         :n-out 500})
               1 (l/dense-layer-builder {:n-in 500
                                         :n-out 100})
               2 (l/output-layer-builder {:loss-fn :negativeloglikelihood
                                          :activation-fn :soft-max
                                          :n-in 100
                                          :n-out output-n})}
      :pretrain false
      :backprop true}))
   (mlb/multi-layer-config-builder {})))

(defn ex-mnist []
  (let [trained-model (-> conf
                            mlb/multi-layer-network
                            u/init
                            (u/ex-train num-epochs train-iterator))
        e (u/new-evaler output-n)
        test-model (u/eval-model trained-model test-iterator e)]
    (u/reset-iterator test-iterator)))

(comment

  (println
   ;; this is conf
   (.build
    (nn-conf/nn-conf-builder
     {:seed rng-seed
      :optimization-algo :stochastic-gradient-descent
      :iterations 1
      :global-activation-fn :relu
      :weight-init :xavier
      :learning-rate l-rate
      :updater :nesterovs
      :momentum 0.98
      :regularization true
      :l2 (* l-rate 0.005)
      :layers {0 (l/dense-layer-builder {:n-in (* num-rows num-cols)
                                         :n-out 500})
               1 (l/dense-layer-builder {:n-in 500
                                         :n-out 100})
               2 (l/output-layer-builder {:loss-fn :negativeloglikelihood
                                          :activation-fn :soft-max
                                          :n-in 100
                                          :n-out output-n})}
      :pretrain false
      :backprop true})))

  (def model (u/init (mlb/multi-layer-network conf)))

  (def trained-model (u/ex-train model num-epochs train-iterator))

  (def evaler (u/new-evaler output-n))

  (def eval-model (u/eval-model trained-model test-iterator evaler)))
