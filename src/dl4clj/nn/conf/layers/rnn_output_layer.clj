(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/RNNOutputLayer.html"}
  dl4clj.nn.conf.layers.rnn-output-layer
  (:require [dl4clj.nn.conf.layers.base-output-layer :as base-out-layer]
            [nd4clj.linalg.lossfunctions.loss-functions :as loss-functions]
            [dl4clj.nn.conf.layers.layer :refer (layer)])
  (:import [org.deeplearning4j.nn.conf.layers RnnOutputLayer RnnOutputLayer$Builder]
           [org.nd4j.linalg.lossfunctions LossFunctions$LossFunction]))
;;deprecated
(defn builder [^RnnOutputLayer$Builder b opts]
  (base-out-layer/builder b opts))

(defn rnn-output [{:keys [loss-function]
                   :or {}
                   :as opts}]
  (.build ^RnnOutputLayer$Builder (builder (RnnOutputLayer$Builder. ^LossFunctions$LossFunction (loss-functions/value-of loss-function)) opts)))

(defmethod layer :rnnoutput [opts]
  (rnn-output (:rnnoutput opts)))


(comment

  (rnn-output {:loss-function :reconstruction-crossentropy})

  (rnn-output {:loss-function :mcxent
               :n-in 100
               :n-out 50
               :activation "softmax"
               :updater :rmsprop
               :weight-init :distribution
               :dist (dl4clj.nn.conf.distribution.uniform-distribution/uniform-distribution -0.08 0.08)
               })

  (layer {:rnnoutput
          {:l1 0.0,
           :drop-out 0.0,
           :custom-loss-function nil,
           :dist {:uniform {:lower -0.08, :upper 0.08}},
           :rho 0.0,
           :activation-function "softmax",
          ;; :learning-rate-after {},
           :gradient-normalization "None",
           :weight-init "DISTRIBUTION",
           :nout 50,
           :adam-var-decay 0.999,
           :bias-init 0.0,
           :lr-score-based-decay 0.0,
           :momentum-after {},
           :loss-function "MCXENT",
           :l2 0.001,
           :updater "RMSPROP",
           :momentum 0.5,
           :layer-name "genisys",
           :nin 100,
           :learning-rate 0.1,
           :adam-mean-decay 0.9,
           :rms-decay 0.95,
           :gradient-normalization-threshold 1.0}})

  )
