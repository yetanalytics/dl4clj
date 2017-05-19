(ns ^{:doc "recreation of: https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/mnist/MLPMnistSingleLayerExample.java"}
    mnst
  (:require [dl4clj.nn.conf.builders.multi-layer-builders :as mlb]
            [dl4clj.nn.conf.builders.nn-conf-builder :as nn-conf]
            [dl4clj.nn.conf.builders.builders :as l]
            [dl4clj.examples.example-utils :as u])
  (:import [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           [org.deeplearning4j.eval Evaluation]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.slf4j Logger]
           [org.slf4j LoggerFactory]))

(def num-rows 28)
(def num-columns 28)
(def output-num 10)
(def batchsize 128)
(def rngseed 123)
(def numepochs 18)
(* 28 28)
(def train-iterator (MnistDataSetIterator. batchsize true rngseed))
(def test-iterator (MnistDataSetIterator. batchsize false rngseed))
(def e (Evaluation. output-num))

(def conf
  (->
   (nn-conf/nn-conf-builder {:seed rngseed
                             :optimization-algo :stochastic-gradient-descent
                             :iterations 1
                             :learning-rate 0.006
                             :updater :nesterovs
                             :momentum 0.9
                             :regularization true
                             :l2 1e-4
                             :layers {0 (l/dense-layer-builder {:n-in (* num-rows num-columns)
                                                                :n-out 1000
                                                                :activation-fn :relu
                                                                :weight-init :xavier})
                                      1 (l/output-layer-builder {:loss-fn :negativeloglikelihood
                                                                 :n-in 1000
                                                                 :n-out output-num
                                                                 :activation-fn :soft-max
                                                                 :weight-init :xavier})}
                             :pretrain false
                             :backprop true})
   (mlb/multi-layer-config-builder {})))


(defn init [^MultiLayerNetwork mln]
  (.init mln)
  mln)

(defn ex-train [^MultiLayerNetwork mln]
  (loop
      [i 0
       result {}]
    (cond (not= i numepochs)
          (do
            (println "current at epoch:" i)
            (recur (inc i)
                   (.fit mln train-iterator)))
          (= i numepochs)
          (do
            (println "training done")
            mln))))

(defn eval-model [mln]
  (while (true? (.hasNext test-iterator))
    (let [nxt (.next test-iterator)
          output (.output mln (.getFeatureMatrix nxt))]
      (do (.eval e (.getLabels nxt) output)
          (println (.stats e))))))

(defn reset-iterator
  [t-iterator]
  (.reset t-iterator))

(defn get-feature-matrix
  [trained-model t-iterator]
  (.data (.output trained-model (.getFeatureMatrix (.next t-iterator)))))

(defn ex-mnist []
  (let [trained-network (-> conf
                            (mlb/multi-layer-network)
                            (u/init)
                            (u/ex-train numepochs train-iterator))
        evaled (eval-model trained-network)]
    (reset-iterator test-iterator)))

(comment

  (ex-mnist)

  (def model (mlb/multi-layer-network conf))

  (def initialized-model (init model))

  (def trained-model (ex-train initialized-model))

  (println
   (str
    (.build
     (nn-conf/nn-conf-builder {:seed rngseed
                               :optimization-algo :stochastic-gradient-descent
                               :iterations 1
                               :learning-rate 0.006
                               :updater :nesterovs
                               :momentum 0.9
                               :regularization true
                               :l2 1e-4
                               :layers {0 (l/dense-layer-builder {:n-in (* num-rows num-columns)
                                                                  :n-out 1000
                                                                  :activation-fn :relu
                                                                  :weight-init :xavier})
                                        1 (l/output-layer-builder {:loss-fn :negativeloglikelihood
                                                                   :n-in 1000
                                                                   :n-out output-num
                                                                   :activation-fn :soft-max
                                                                   :weight-init :xavier})}
                               :pretrain false
                               :backprop true}))))
  )
