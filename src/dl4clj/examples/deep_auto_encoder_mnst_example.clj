(ns dl4clj.examples.deep-auto-encoder-mnst-example
  (:import [org.deeplearning4j.datasets.fetchers MnistDataFetcher]
           [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.deeplearning4j.optimize.api IterationListener]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.api Layer]
           [org.deeplearning4j.eval Evaluation]
           [org.deeplearning4j.nn.api Model])
  (:require [dl4clj.nn.conf.builders.nn-conf-builder :as nn]
            [dl4clj.nn.conf.builders.builders :as l]
            [dl4clj.examples.example-utils :as u]
            [dl4clj.nn.conf.builders.multi-layer-builders :as mlb]))

;; not tuned but might be worth working on if time
(comment
  (def num-rows 28)
  (def num-cols 28)
  (def seed 123)
  (def num-samples (.totalExamples (MnistDataFetcher. )))
  (def batch-size 1000)
  (def iterations 1)
  (def listener-freq (/ iterations 5))

  (def iter (MnistDataSetIterator. batch-size num-samples true))

  (def conf
    (.build (nn/nn-conf-builder {:seed seed
                                 :iterations iterations
                                 :optimization-algo :line-gradient-descent
                                 :layers {0 {:rbm {:n-in (* num-rows num-cols)
                                                   :n-out 1000
                                                   :loss-fn :kl-divergence}}
                                          1 {:rbm {:n-in 1000
                                                   :n-out 500
                                                   :loss-fn :kl-divergence}}
                                          2 {:rbm {:n-in 500
                                                   :n-out 250
                                                   :loss-fn :kl-divergence}}
                                          3 {:rbm {:n-in 250
                                                   :n-out 100
                                                   :loss-fn :kl-divergence}}
                                          4 {:rbm {:n-in 100
                                                   :n-out 30
                                                   :loss-fn :kl-divergence}}
                                          5 {:rbm {:n-in 30
                                                   :n-out 100
                                                   :loss-fn :kl-divergence}}
                                          6 {:rbm {:n-in 100
                                                   :n-out 250
                                                   :loss-fn :kl-divergence}}
                                          7 {:rbm {:n-in 250
                                                   :n-out 500
                                                   :loss-fn :kl-divergence}}
                                          8 {:rbm {:n-in 500
                                                   :n-out 1000
                                                   :loss-fn :kl-divergence}}
                                          9 {:output-layer {:loss-fn :mse
                                                            :activation-fn :sigmoid
                                                            :n-in 1000
                                                            :n-out (* num-rows num-cols)}}}
                                 :pretrain true
                                 :backprop true})))
  #_(clojure.pprint/pprint (sort-by :name (filter :exception-types (:members (clojure.reflect/reflect model)))))

  (def mln (mlb/multi-layer-network conf))

  (def model (u/init mln))

  (defn training-model
    [model iterator]
    (while (true? (.hasNext iterator))
      (let [nxt (.next iterator)]
        (.fit model (DataSet. (.getFeatureMatrix nxt) (.getFeatureMatrix nxt)))))
    model)
  (def trained-model (training-model model iter))

  ;; eval is being funky, moving onto another example for now
  )
