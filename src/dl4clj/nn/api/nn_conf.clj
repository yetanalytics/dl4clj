(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/NeuralNetConfiguration.html"}
  dl4clj.nn.api.nn-conf
  (:import [org.deeplearning4j.nn.conf NeuralNetConfiguration])
  (:require [clojure.core.match :refer [match]]))

;; left off updating API fns here
(defn add-variable!
  ":var-name (str), the name of the variable you want to add to the nn-conf"
  [& {:keys [nn-conf var-name]
      :as opts}]
  (match [opts]
         [{:nn-conf (_ :guard seq?)
           :var-name (:or (_ :guard string?)
                          (_ :guard seq?))}]
         `(doto ~nn-conf (.addVariable ~var-name))
         :else
         (doto nn-conf (.addVariable var-name))))

(defn clear-variables!
  [nn-conf]
  (match [nn-conf]
         [(_ :guard seq?)]
         `(doto ~nn-conf .clearVariables)
         :else
         (doto nn-conf .clearVariables)))

(defn from-json
  [json]
  (match [json]
         [(_ :guard seq?)]
         `(NeuralNetConfiguration/fromJson ~json)
         :else
         (NeuralNetConfiguration/fromJson json)))

(defn from-yaml
  [json]
  (match [json]
         [(_ :guard seq?)]
         `(NeuralNetConfiguration/fromYaml ~json)
         :else
         (NeuralNetConfiguration/fromYaml json)))

(defn get-l1-by-param
  [& {:keys [nn-conf var-name]
      :as opts}]
  (match [opts]
         [{:nn-conf (_ :guard seq?)
           :var-name (:or (_ :guard string?)
                          (_ :guard seq?))}]
         `(.getL1ByParam ~nn-conf ~var-name)
         :else
         (.getL1ByParam nn-conf var-name)))

(defn get-l2-by-param
  [& {:keys [nn-conf var-name]
      :as opts}]
  (match [opts]
         [{:nn-conf (_ :guard seq?)
           :var-name (:or (_ :guard string?)
                          (_ :guard seq?))}]
         `(.getL2ByParam ~nn-conf ~var-name)
         :else
         (.getL2ByParam nn-conf var-name)))

(defn get-learning-rate-by-param
  [& {:keys [nn-conf var-name]
      :as opts}]
  (match [opts]
         [{:nn-conf (_ :guard seq?)
           :var-name (:or (_ :guard string?)
                          (_ :guard seq?))}]
         `(.getLearningRateByParam ~nn-conf ~var-name)
         :else
         (.getLearningRateByParam nn-conf var-name)))

(defn mapper
  "Object mapper for serialization of configurations"
  [nn-conf]
  (match [nn-conf]
         [(_ :guard seq?)]
         `(.mapper ~nn-conf)
         :else
         (.mapper nn-conf)))

(defn mapper-yaml
  "Object mapper for serialization of configurations"
  [nn-conf]
  (match [nn-conf]
         [(_ :guard seq?)]
         `(.mapperYaml ~nn-conf)
         :else
         (.mapperYaml nn-conf)))

(defn reinit-mapper-with-subtypes
  "Reinitialize and return the Jackson/json ObjectMapper with additional named types.

  typez (coll), a collection of json named types"
  [typez]
  (match [typez]
         [(_ :guard seq?)]
         `(.reinitMapperWithSubtypes ~typez)
         :else
         (.reinitMapperWithSubtypes typez)))

(defn set-layer-param-lr!
  "sets the layer learning rate for the variable and returns the nn-conf"
  [& {:keys [nn-conf var-name]
      :as opts}]
  (match [opts]
         [{:nn-conf (_ :guard seq?)
           :var-name (:or (_ :guard string?)
                          (_ :guard seq?))}]
         `(doto ~nn-conf (.setLayerParamLR ~var-name))
         :else
         (doto nn-conf (.setLayerParamLR var-name))))

(defn set-learning-rate-by-param!
  ":rate (double), the learning rate for the variable supplied

  returns the nn-conf"
  [& {:keys [nn-conf var-name rate]
      :as opts}]
  (match [opts]
         [{:nn-conf (_ :guard seq?)
           :var-name (:or (_ :guard string?)
                          (_ :guard seq?))
           :rate (:or (_ :guard number?)
                      (_ :guard seq?))}]
         `(doto ~nn-conf (.setLearningRateByParam ~var-name (double ~rate)))
         :else
         (doto nn-conf (.setLearningRateByParam var-name rate))))

(defn list-variables
  [& {:keys [nn-conf copy?]
      :as opts}]
  (match [opts]
         [{:nn-conf (_ :guard seq?)
           :copy? (:or (_ :guard boolean?)
                       (_ :guard seq?))}]
         `(.variables ~nn-conf ~copy?)
         [{:nn-conf _
           :copy? _}]
         (.variables nn-conf copy?)
         [{:nn-conf (_ :guard seq?)}]
         `(.variables ~nn-conf)
         :else
         (.variables nn-conf)))

(defn to-json
  [nn-conf]
  (match [nn-conf]
         [(_ :guard seq?)]
         `(.toJson ~nn-conf)
         :else
         (.toJson nn-conf)))

(defn build-nn
  [nn-conf]
  (match [nn-conf]
         [(_ :guard seq?)]
         `(.build ~nn-conf)
         :else
         (.build nn-conf)))
