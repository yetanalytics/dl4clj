(ns dl4clj.clustering.cluster.point
  (:import [org.deeplearning4j.clustering.cluster Point])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn new-point
  [& {:keys [id label data]
      :as opts}]
  (assert (contains? opts :data)
          "you must provided point data in the form of a double array or as an NDArray")
  ;; just have users pass data as a clojure vector
  ;; build conversion into type checking multimethod in util
  (cond (contains-many? opts :data :label :id)
        (Point. id label data)
        (contains-many? opts :id data)
        (Point. id data)
        :else
        (Point. data)))

(defn get-array
  [point]
  (.getArray point))

(defn get-id
  [point]
  (.getId point))

(defn get-label
  [point]
  (.getLabel point))

(defn set-data!
  [& {:keys [point data]}]
  (doto point (.setArray data)))

(defn set-id!
  [& {:keys [point id]}]
  (doto point (.setId id)))

(defn set-label!
  [& {:keys [point label]}]
  (doto point (.setLabel label)))

(defn to-points
  [list-of-vecs]
  (.toPoints list-of-vecs))
