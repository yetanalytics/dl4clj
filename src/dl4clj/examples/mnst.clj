(ns mnst
  (:require [dl4clj.nn.conf.multi-layer-configuration :as l-builder])
  (:import [org.nd4j.linalg.dataset.api.iterator DataSetIterator]
           [org.nd4j.linalg.activations.Activations ]
           [org.deeplearning4j.datasets.iterator.impl MnistDataSetIterator]
           [org.deeplearning4j.eval Evaluation]
           [org.deeplearning4j.nn.api OptimizationAlgorithm]
           [org.deeplearning4j.nn.conf MultiLayerConfiguration MultiLayerConfiguration$Builder]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration NeuralNetConfiguration$Builder]
           [org.deeplearning4j.nn.conf Updater]
           [org.deeplearning4j.nn.conf.layers DenseLayer DenseLayer$Builder]
           [org.deeplearning4j.nn.conf.layers OutputLayer OutputLayer$Builder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]
           [org.deeplearning4j.nn.weights WeightInit]
           [org.deeplearning4j.optimize.listeners ScoreIterationListener]
           [org.nd4j.linalg.api.ndarray INDArray]
           [org.nd4j.linalg.dataset DataSet]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]
           [org.slf4j Logger]
           [org.deeplearning4j.nn.conf NeuralNetConfiguration$ListBuilder]
           [org.deeplearning4j.nn.conf.layers Layer$Builder]
           [org.slf4j LoggerFactory]))

;; make a ns for all the builders and name the defns appropriately

(def num-rows 28)
(def num-columns 28)
(def output-num 10)
(def batchsize 128)
(def rngspeed 123)
(def numepochs 15)

(def train-iterator (MnistDataSetIterator. batchsize true rngspeed))
(def test-iterator (MnistDataSetIterator. batchsize false rngspeed))


(defn dense-layer-builder
  ([]
   (dense-layer-builder (DenseLayer$Builder.) {}))
  ([opts]
   (dense-layer-builder (DenseLayer$Builder.) opts))
  ([^DenseLayer$Builder dense-layer-builder {:keys [n-in
                                                    n-out
                                                    activation-fn
                                                    weight-init
                                                    ]
                                 :or {}
                                 :as opts}]
   (when (or n-in (contains? opts :n-in))
     (.nIn dense-layer-builder (* num-rows num-columns)))
   (when (or n-out (contains? opts :n-out))
     (.nOut dense-layer-builder 1000))
   (when (or activation-fn (contains? opts :activation-fn))
     (.activation dense-layer-builder "RELU"))
   (when (or weight-init (contains? opts :weight-init))
     (.weightInit dense-layer-builder (WeightInit/XAVIER)))
   (.build dense-layer-builder)))

(dense-layer-builder {:n-in 5
                      :n-out 1000
                      :activation-fn "RELU"
                      :weight-init :XAVIER
                      :layer-n 0})

(defn output-layer-builder
  ([]
   (output-layer-builder (OutputLayer$Builder.) {}))
  ([opts]
   (output-layer-builder (OutputLayer$Builder.) opts))
  ([^OutputLayer$Builder output-layer-builder {:keys [loss-fn?
                                                      n-in
                                                      n-out
                                                      activation-fn
                                                      weight-init]
                                               :or {}
                                               :as opts}]
   (when (or loss-fn? (contains? opts :loss-fn?))
     (.lossFunction output-layer-builder (LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD)))
   (when (or n-in (contains? opts :n-in))
     (.nIn output-layer-builder 1000))
   (when (or n-out (contains? opts :n-out))
     (.nOut output-layer-builder 10))
   (when (or activation-fn (contains? opts :activation-fn))
     (.activation output-layer-builder "SOFTMAX"))
   (when (or weight-init (contains? opts :weight-init))
     (.weightInit output-layer-builder (WeightInit/XAVIER)))
   (.build output-layer-builder)))

(output-layer-builder {:loss-fn? "foo"
                       :n-in 1000
                       :n-out 10
                       :activation-fn "SOFTMAX"
                       :weight-init "XAVIER"})
;; list builder
;; multi layer configuration builder

(defn builder
  [{:keys [rngseed
           optimizer
           iterations
           l-rate
           updater
           reg
           l?
           layer-1
           layer-2
           pretrain?]
    :or {}
    :as opts}]
  (let [b (NeuralNetConfiguration$Builder.)]
   (when (or rngseed (contains? opts :rngseed))
    (.seed b (long rngseed)))
  (when (or optimizer (contains? opts :optimizer))
    (.optimizationAlgo b (OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)))
  (when (or iterations (contains? opts :iterations))
    (.iterations b 1))
  (when (or l-rate (contains? opts :l-rate))
    (.learningRate b 0.006))
  (when (or updater (contains? opts :updater))
    (-> (.updater b (Updater/NESTEROVS))
        (.momentum 0.9)))
  (when (or reg (contains? opts :reg))
    (-> (.regularization b true)
        (.l2 1e-4)))
  (when (or l? (contains? opts :l?))
    (.list b))
  (when (or pretrain? (contains? opts :pretrain?))
    (-> (.pretrain b false)
        (.backprop true)))
  (when (or layer-1 (contains? opts :layer-1))
    (.layer ;; list builder
     b
     #_(list builder 0 (dense-layer-builder {:n-in 10
                                             :n-out 10
                                             :activation-fn "foo"
                                             :weight-init "sup"}))
     (dense-layer-builder {:n-in 10
                           :n-out 10
                           :activation-fn "foo"
                           :weight-init "sup"})

            )
    #_(dense-layer-builder {:n-in 10
                          :n-out 10
                          :activation-fn "foo"
                          :weight-init "sup"})
    #_(output-layer-builder {:loss-fn? "foo"
                           :n-in 1000
                           :n-out 10
                           :activation-fn "SOFTMAX"
                             :weight-init "XAVIER"}))
  #_(when (or layer-2 (contains? opts :layer-2))
    (.layer b 1 (output-layer-builder {:loss-fn? "foo"
                                   :n-in 1000
                                   :n-out 10
                                   :activation-fn "SOFTMAX"
                                   :weight-init "XAVIER"})
           ))
  (.build b)))

#_(when (or layer-2 (contains? opts :layer-2))
  (.layer b (output-layer-builder {:loss-fn? "foo"
                                   :n-in 1000
                                   :n-out 10
                                   :activation-fn "SOFTMAX"
                                   :weight-init "XAVIER"})
          ))

#_(when (or layer-1 (contains? opts :layer-1))
  (.layer b (dense-layer-builder {:n-in 10
                                  :n-out 10
                                  :activation-fn "foo"
                                  :weight-init "sup"})
          ))
;; multi layer and list builder go hand in hand
;; multilayer builder config

(defn init [^MultiLayerNetwork mln]
  (.init mln)
  mln)

(defn ex-train [^MultiLayerNetwork mln]
  (loop
      [i 0
       stop numepochs]
    (cond (not= i stop)
          (recur (inc i)
                 (.fit mln train-iterator))
          (= i stop)
          mln)))

(ex-train
 (init
  (builder
   (MultiLayerNetwork.
   (multi-layer-configuration-builder
    (builder {:rngseed 123
              :optimizer "s"
              :iterations 1
              :l-rate 0.006
              :updater 0.9
              :reg 1e-4
              :l? true
              :layer-1 true
              :layer-2 true
              :pretrain true
              }))))))
