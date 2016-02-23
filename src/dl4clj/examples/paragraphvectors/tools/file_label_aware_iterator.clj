(ns ^{:doc "see https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/paragraphvectors/tools/FileLabelAwareIterator.java"}
  dl4clj.examples.paragraphvectors.tools.file-label-aware-iterator
  (:require [dl4clj.text.documentiterator.labels-source :refer (labels-source)]
            [dl4clj.text.documentiterator.labelled-document :refer (labelled-document)])
  (:import [org.deeplearning4j.text.documentiterator LabelAwareIterator]))

(defrecord FileLabelAwareIterator [docs position labels-source]
  LabelAwareIterator
  (getLabelsSource [this]
    labels-source)
  (hasNextDocument [this]
    (< @position (count docs)))
  (nextDocument [this]
    (try (let [file-to-read ^java.io.File (nth docs @position)]
           (labelled-document (slurp file-to-read)
                                    (.getName (.getParentFile file-to-read))))
         (catch Exception e
           (throw (RuntimeException. e)))
         (finally (swap! position inc))))
  (reset [this]
    (reset! position 0)))

(defn builder 
  ([& root-folders]
   (atom (map clojure.java.io/as-file root-folders))))

(defn add-source-folder [builder folder]
  (swap! builder conj (clojure.java.io/as-file folder)))

(defn build [b]
  (let [root-dirs (filter #(.isDirectory ^java.io.File %) @b)
        labels (reduce conj #{}
                       (mapcat (fn [^java.io.File root-dir]
                                 (map #(.getName ^java.io.File %)
                                      (filter #(and (.isDirectory ^java.io.File %)
                                                    (not-empty (.listFiles ^java.io.File %)))
                                              (.listFiles root-dir)))) 
                               root-dirs))
        docs (mapcat (fn [^java.io.File root-dir]
                       (mapcat (fn [label-dir]
                                 (remove #(.isDirectory ^java.io.File %)
                                         (.listFiles ^java.io.File label-dir)))
                               (.listFiles root-dir)))
                     root-dirs)]
    (FileLabelAwareIterator. docs (atom 0) (labels-source (into [] labels)))))

(defn file-label-aware-iterator [& root-folders]
  (build (apply builder root-folders)))


(comment
  
  (def b (builder (clojure.java.io/resource "paravec/labeled")))
  (def iter (build b))

  (def iter (file-label-aware-iterator (clojure.java.io/resource "paravec/labeled")))

)
