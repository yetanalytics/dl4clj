(ns dl4clj.nn.conf.multi-layer-builders
  (:require [dl4clj.nn.conf.builders :as b])
  (:import [org.deeplearning4j.nn.conf #_NeuralNetConfiguration NeuralNetConfiguration$Builder
            NeuralNetConfiguration$ListBuilder #_MultiLayerConfiguration MultiLayerConfiguration$Builder]))


(defn multi-layer-config-builder
  ([]
   (multi-layer-config-builder (MultiLayerConfiguration$Builder.) {}))
  ([opts]
   (multi-layer-config-builder (MultiLayerConfiguration$Builder.) {}))
  ([^MultiLayerConfiguration$Builder b {:keys [confs]
                                        :or {}
                                        :as opts}]
   (if (contains? opts :confs)
     (.confs b confs))
   b))

(defn list-builder [nn-conf-builder layers opts]
  (let [b (NeuralNetConfiguration$ListBuilder. nn-conf-builder)]
    (doseq [[idx l] (:layers layers)]
      (.layer b idx (b/builder l)))
    b))


#_(list-builder (NeuralNetConfiguration$Builder.) {:layers {0 {:graves-lstm {:layer-name "first layer"}}
                                                          1 {:graves-lstm {:layer-name "genisys"}}}}
              {})
