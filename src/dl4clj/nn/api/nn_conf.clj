(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/NeuralNetConfiguration.html"}
  dl4clj.nn.api.nn-conf
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration]))

(defn add-variable!
  ":var-name (str), the name of the variable you want to add to the nn-conf"
  [& {:keys [nn-conf var-name]}]
  (doto nn-conf (.addVariable var-name)))

(defn clear-variables!
  [nn-conf]
  (doto nn-conf (.clearVariables)))

(defn from-json
  [json]
  (NeuralNetConfiguration/fromJson json))

(defn from-yaml
  [json]
  (NeuralNetConfiguration/fromYaml json))

(defn get-l1-by-param
  [& {:keys [nn-conf var-name]}]
  (.getL1ByParam nn-conf var-name))

(defn get-l2-by-param
  [& {:keys [nn-conf var-name]}]
  (.getL2ByParam nn-conf var-name))

(defn get-learning-rate-by-param
  [& {:keys [nn-conf var-name]}]
  (.getLearningRateByParam nn-conf var-name))

(defn mapper
  "Object mapper for serialization of configurations"
  [nn-conf]
  (.mapper nn-conf))

(defn mapper-yaml
  "Object mapper for serialization of configurations"
  [nn-conf]
  (.mapperYaml nn-conf))

(defn reinit-mapper-with-subtypes
  "Reinitialize and return the Jackson/json ObjectMapper with additional named types.

  typez (coll), a collection of json named types"
  [typez]
  (.reinitMapperWithSubtypes typez))

(defn set-layer-param-lr!
  "sets the layer learning rate for the variable and returns the nn-conf"
  [& {:keys [nn-conf var-name]}]
  (doto nn-conf (.setLayerParamLR var-name)))

(defn set-learning-rate-by-param!
  ":rate (double), the learning rate for the variable supplied

  returns the nn-conf"
  [& {:keys [nn-conf var-name rate]}]
  (doto nn-conf (.setLearningRateByParam var-name rate)))

(defn list-variables
  [& {:keys [nn-conf copy?]}]
  (if (boolean? copy?)
    (.variables nn-conf copy?)
    (.variables nn-conf)))

(defn to-json
  [nn-conf]
  (.toJson nn-conf))

(defn build-nn
  [nn-conf]
  (.build nn-conf))
