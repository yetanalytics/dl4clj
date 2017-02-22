(ns dl4clj.nn.conf.builders.multi-layer-builders
  (:require [dl4clj.nn.conf.builders :as b]
            [dl4clj.nn.conf.backprop-type :as backprop-t]
            [dl4clj.nn.conf.builders.nn-conf-builder :as nn])
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration$Builder
            NeuralNetConfiguration$ListBuilder MultiLayerConfiguration$Builder]
           [org.deeplearning4j.nn.multilayer MultiLayerNetwork]))

(defn multi-layer-config-builder
  ([]
   (multi-layer-config-builder (MultiLayerConfiguration$Builder.) {}))
  ([opts]
   (multi-layer-config-builder (MultiLayerConfiguration$Builder.) opts))
  ([^MultiLayerConfiguration$Builder multi-layer-config-b
    {:keys [backprop ;; boolean
            backprop-type ;; (one of (backprop-type/values))
     ;;       confs ;; java.util.List<NeuralNetConfiguration>
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
     (.backpropType multi-layer-config-b (backprop-t/value-of backprop-type))
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
   multi-layer-config-b
   #_(.build multi-layer-config-b)))

(defn list-builder [nn-conf-builder layers]
  (let [b (.list nn-conf-builder)
        #_(NeuralNetConfiguration$ListBuilder. nn-conf-builder)]
    (doseq [[idx l] (:layers layers)]
      (.layer b idx (b/builder l)))
    b))

#_(.build (list-builder (NeuralNetConfiguration$Builder.)
              {:layers {0 {:graves-lstm {:layer-name "first layer"}}
                        1 {:graves-lstm {:layer-name "genisys"}}}}
              ))

#_(list-builder (MultiLayerConfiguration$Builder.) ;; cant do this
              {:layers {0 {:graves-lstm {:layer-name "first layer"}}
                        1 {:graves-lstm {:layer-name "genisys"}}}}
              )
;; this doesnt work because there are now two builders of the same type just nested at different levels
(.confs (MultiLayerConfiguration$Builder.)
        (list (list-builder (NeuralNetConfiguration$Builder.)
                            {:layers {0 {:graves-lstm {:layer-name "first layer"}}
                                      1 {:graves-lstm {:layer-name "genisys"}}
                                      }}
                            )))

;; this is not possible any more
(.layer (multi-layer-config-builder (.list (NeuralNetConfiguration$Builder.)) {:backprop true}))

(-> (NeuralNetConfiguration$Builder.)
    (.iterations 2)
    (.seed 123)
    #_(.list)
    #_(type)
    #_(.layer 0 (b/builder {:graves-lstm {:layer-name "first layer"}}))
    #_(.layer 1 (b/builder {:graves-lstm {:layer-name "second layer"}}))
    #_(.layer (b/builder {:graves-lstm {:layer-name "first layer"}})) ;; this sets the layer but only one of them
    (list-builder {:layers {0 {:graves-lstm {:layer-name "first layer"
                                             :n-in 10
                                             :n-out 10}}
                            1 {:graves-lstm {:layer-name "genisys"
                                             :n-in 10
                                             :n-out 20}}
                            }})
    #_(type)
    #_(.getLayer )
    #_(.pretrain false)
    #_(.backprop true)
    #_(type)
    #_(.backprop true)
    #_(.tBPTTBackwardLength 10)
    (multi-layer-config-builder {:backprop false
                                 :tbptt-back-length 10
                                 :tbptt-fwd-length 30
                                 :pretrain false})
    #_(type)
    (.build)
    #_(type)
    #_(MultiLayerNetwork.)
    #_(.epsilon 0.2)
    #_(.init)
    #_(.build)
    )

;; this is working
(-> (nn/nn-conf-builder {:drop-out 2
                         :backprop true})
    (list-builder {:layers {0 {:graves-lstm {:layer-name "first layer"
                                             :n-in 10
                                             :n-out 10
                                             :epsilon 2.0}}
                            1 {:graves-lstm {:layer-name "genisys"
                                             :n-in 10
                                             :n-out 20}}
                            }})
    (multi-layer-config-builder {:backprop false
                                 :tbptt-back-length 10
                                 :tbptt-fwd-length 30
                                 :pretrain false})
    (.build)
    (str)
    #_(clojure.pprint/pprint)
    )




;; this seems like the correct process
(.fit (MultiLayerNetwork. (.build (list-builder  (NeuralNetConfiguration$Builder.)
                                                  {:layers {0 {:dense-layer {:n-in 100
                                                                             :n-out 1000
                                                                             :layer-name "first layer"}}
                                                            1 {:output-layer {:layer-name "second layer"
                                                                              :n-in 1000
                                                                              :n-out 10
                                                                              }}}}
                     ))))
(defn t [b]
  (->
   (nn/nn-conf-builder b)
   (list-builder {:layers {0 {:dense-layer {:n-in 100
                                            :n-out 1000
                                            :layer-name "first layer"}}
                           1 {:output-layer {:layer-name "second layer"
                                             :n-in 1000
                                             :n-out 10
                                             }}}} )
   #_(.build )
   (multi-layer-config-builder {:backprop true})))

(MultiLayerNetwork. (.build (t {:seed 123})))

;; bellow shows that the layers still arent being set properly
(.getLayer (MultiLayerNetwork. (.build (t (NeuralNetConfiguration$Builder.)))) 0)
