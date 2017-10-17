(ns dl4clj.clustering.cluster.point
  (:import [org.deeplearning4j.clustering.cluster Point])
  (:require [dl4clj.utils :refer [contains-many?]]
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn new-point
  [& {:keys [id label data]
      :as opts}]
  (let [d (vec-or-matrix->indarray data)]
   (cond (contains-many? opts :data :label :id)
        (Point. id label d)
        (contains-many? opts :id :data)
        (Point. id d)
        :else
        (Point. d))))

(defn get-point-data
  [point]
  (.getArray point))

(defn get-point-id
  [point]
  (.getId point))

(defn get-point-label
  [point]
  (.getLabel point))

(defn set-point-data!
  [& {:keys [point data]}]
  (doto point (.setArray (vec-or-matrix->indarray data))))

(defn set-point-id!
  [& {:keys [point id]}]
  (doto point (.setId id)))

(defn set-point-label!
  [& {:keys [point label]}]
  (doto point (.setLabel label)))

(defn to-points
  ;; refactor/understand
  [list-of-vecs]
  (.toPoints list-of-vecs))
