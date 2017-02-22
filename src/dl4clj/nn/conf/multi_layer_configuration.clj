(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/MultiLayerConfiguration.html"}
  dl4clj.nn.conf.multi-layer-configuration
  (:require [dl4clj.nn.conf.backprop-type :as backprop-type]
            [dl4clj.nn.conf.layers.layer :as layer]
            [clojure.data.json :as json]
            [dl4clj.utils :refer (camel-to-dashed)])
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration$ListBuilder]
           [org.deeplearning4j.nn.conf MultiLayerConfiguration MultiLayerConfiguration$Builder]))

(defn builder
  ([]
   (builder (MultiLayerConfiguration$Builder.) {}))
  ([opts]
   (builder (MultiLayerConfiguration$Builder.) opts))
  ([^MultiLayerConfiguration$Builder builder {:keys [backprop ;; Whether to do back prop or not (boolean)
                                                     backprop-type ;; (one of (backprop-type/values))
                                                     cnn-input-size ;; CNN input size, in order of [height,width,depth] (int-array)
                                                     confs ;; java.util.List<NeuralNetConfiguration>
                                                     input-pre-processors ;; ({integer,InputPreProcessor})
                                                     pretrain ;; Whether to do pre train or not (boolean)
                                                     redistribute-params ;; Whether to redistribute parameters as a view or not (boolean)
                                                     t-bptt-backward-length ;; When doing truncated BPTT: how many steps of backward should we do?
                                                     ;; Only applicable when doing backpropType(BackpropType.TruncatedBPTT)
                                                     ;; This is the k2 parameter on pg23 of http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf(int)
                                                     t-bptt-forward-length ;; When doing truncated BPTT: how many steps of forward pass should we do before doing (truncated) backprop? (int)
                                                     ;; Only applicable when doing backpropType(BackpropType.TruncatedBPTT)
                                                     ;; Typically tBPTTForwardLength parameter is same as the the tBPTTBackwardLength parameter, but may be larger than it in some circumstances (but never smaller)
                                                     ;; Ideally your training data time series length should be divisible by this This is the k1 parameter on pg23 of http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
                                                     ]
                                              :or {}
                                              :as opts}]
   (when (or backprop (contains? opts :backprop))
     (.backprop builder backprop))
   (when backprop-type
     (.backpropType builder (backprop-type/value-of backprop-type)))
   (when cnn-input-size
     (.cnnInputSize builder (int-array cnn-input-size)))
   (when confs
     (.confs builder confs))
   (when input-pre-processors
     (.inputPreProcessors builder input-pre-processors))
   (when (or pretrain (contains? opts :pretrain))
     (.pretrain builder pretrain))
   (when (or redistribute-params (contains? opts :redistribute-params))
     (.redistributeParams builder redistribute-params))
   (when t-bptt-backward-length
     (.tBPTTBackwardLength builder t-bptt-backward-length))
   (when t-bptt-forward-length
     (.tBPTTForwardLength builder t-bptt-forward-length))
   builder))


(defn list-builder [^NeuralNetConfiguration$ListBuilder b layers opts]
  (let [b (builder b opts)]
    (doseq [[idx l] layers]
      (.layer b idx (if (map? l) (layer/layer l) l)))
    b))

(defn to-json [^MultiLayerConfiguration cfg]
  (.toJson cfg))
(defn from-json [^String  cfg]
  (MultiLayerConfiguration/fromJson cfg))
(defn to-edn [^MultiLayerConfiguration cfg]
  (json/read-str (.toJson cfg)
                 :key-fn #(keyword (camel-to-dashed %))))


;; (defn from-edn [cfg]
;;   (.build ^MultiLayerConfiguration$Builder (builder (update-in cfg [:confs] #(map neural-net-configuration %)))))
#_(builder {:optimization-algo :stochastic-gradient-descent
             :iterations 1
             :learning-rate 0.1
             :rms-decay 0.95
             :seed 12345
             :regularization true
             :l2 0.001
             :list 3
          :confs  (list {0 {:graves-lstm {:n-in 50
                                    :n-out 100
                                    :updater :rmsprop
                                    :activation :tanh
                                    :weight-init :distribution
                                    :dist {:binomial {:number-of-trials 0, :probability-of-success 0.08}}}}
                   1 {:graves-lstm {:n-in 100
                                    :n-out 100
                                    :updater :rmsprop
                                    :activation :tanh
                                    :weight-init :distribution
                                    :dist {:uniform {:lower -0.08, :upper 0.08}}}}
                   2 {:rnnoutput {:loss-function :mcxent
                                  :n-in 100
                                  :n-out 50
                                  :activation :softmax
                                  :updater :rmsprop
                                  :weight-init :distribution
                                  :dist {:normal {:mean 0.0, :std 0.05}}}}})
             :pretrain false
             :backprop true})
