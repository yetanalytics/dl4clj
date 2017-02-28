(ns dl4clj.nn.conf.builders.multi-layer-builders
  (:require [dl4clj.nn.conf.builders.builders :as bb]
            [dl4clj.nn.conf.backprop-type :as backprop-t]
            [dl4clj.nn.conf.constants :as constants])
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder
            NeuralNetConfiguration$ListBuilder MultiLayerConfiguration$Builder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]))
;; update docs

(defn multi-layer-config-builder
  "creates a multi layer neural network configuration to be used within a multilayernetwork.

  params are:

  :backprop (boolean) whether to do backprop or not

  :backprop-type (keyword) the type of backprop, one of :standard or :truncated-bptt

  :input-pre-processors [int keyword] ^one layer or {int keyword} ^multiple layers
  specifies the processors, these are used at each layer for doing things like
  ormalization and shaping of input.
"
  ([]
   (multi-layer-config-builder (MultiLayerConfiguration$Builder.) {}))
  ([opts]
   (multi-layer-config-builder (MultiLayerConfiguration$Builder.) opts))
  ([^MultiLayerConfiguration$Builder multi-layer-config-b
    {:keys [backprop ;; boolean
            backprop-type ;; (one of (backprop-type/values))
            ;; confs java.util.List<NeuralNetConfiguration>
            ;; confs is automaticaly set when you .build
            damping-factor ;; double
            input-pre-processors ;; ({integer,InputPreProcessor})
            input-type ;; Convolutional/ff/recurrent
            ;;see "https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/inputs/InputType.html"
            pretrain ;; boolean
            tbptt-back-length ;; int
            tbptt-fwd-length ;; int
            ]
     :or {}
     :as opts}]
   (if (contains? opts :backprop)
     (.backprop multi-layer-config-b backprop) multi-layer-config-b)
   (if (contains? opts :backprop-type)
     (.backpropType multi-layer-config-b
                    (constants/value-of {:backprop-type backprop-type}))
     multi-layer-config-b)
   #_(if (contains? opts :confs)
     (.confs multi-layer-config-b confs) multi-layer-config-b)
   (if (contains? opts :damping-factor)
     (.dampingFactor multi-layer-config-b damping-factor) multi-layer-config-b)
   (if (contains? opts :input-pre-processors)
     (.inputPreProcessors multi-layer-config-b input-pre-processors)
     multi-layer-config-b)
   (if (contains? opts :input-type)
     (.inputType multi-layer-config-b input-type) multi-layer-config-b)
   (if (contains? opts :pretrain)
     (.pretrain multi-layer-config-b pretrain) multi-layer-config-b)
   (if (contains? opts :tbptt-back-length)
     (.tBPTTBackwardLength multi-layer-config-b tbptt-back-length)
     multi-layer-config-b)
   (if (contains? opts :tbptt-fwd-length)
     (.tBPTTForwardLength multi-layer-config-b tbptt-fwd-length)
     multi-layer-config-b)
   multi-layer-config-b))

(defn list-builder [nn-conf-builder layers]
  (let [b (.list nn-conf-builder)]
    (doseq [[idx l] layers]
      (.layer b idx (bb/builder l)))
    b))

(comment
;; this is working
  (-> (dl4clj.nn.conf.builders.nn-conf-builder/nn-conf-builder
       {:drop-out 2
        :backprop true
        :seed 123
        :global-activation-fn "CUBE" ;; wont overwrite the activation fns set at the layer level
        :layers {0 {:graves-lstm {:layer-name "first layer"
                                  :n-in 10
                                  :n-out 10
                                  :activation-fn "RELU"
                                  :epsilon 2.0}}
                 1 {:graves-lstm {:layer-name "genisys"
                                  :activation-fn "SOFTMAX"
                                  :n-in 10
                                  :n-out 20}}}})
      (multi-layer-config-builder {:backprop true
                                   :tbptt-back-length 10
                                   :tbptt-fwd-length 30
                                   :pretrain false})
      (.build)
      )
  ;; will need to add checks to make sure the ovlapping config maps for nn-conf and multi layer are the same
  ;; need error handling which determines if there are :layer and :layers keys and warns that :layer will be lost
  ;; that detection will also determine if the output of nn-conf-builder will be set to multi-layer-conf-builder or not
  )
