(ns ^{:doc "
A DataSetIterator for use in the graves-lstm-char-modelling-example

@author Joachim De Beule
"}
  dl4clj.examples.rnn.lr-character-iterator
  (:require [nd4clj.linalg.factory.nd4j :refer (zeros)]
            [dl4clj.examples.example-utils :refer (shakespeare index-map)]
            [nd4clj.linalg.api.ndarray.indarray :refer (put-scalar get-scalar shape)]
            [nd4clj.linalg.dataset.api.data-set :refer :all]
            [nd4clj.linalg.dataset.data-set :refer (data-set)])
  (:import [java.util NoSuchElementException]           
           [org.deeplearning4j.datasets.iterator DataSetIterator]))

(defrecord LRCharDataSetIterator [valid-chars char-to-idx ^"[C" input-chars segment-length n-segments batch-size char-pointer max-char-pointer]
  DataSetIterator
  (batch [_] batch-size)
  (cursor [_] (mod @char-pointer segment-length)) 
  (inputColumns [_] (count valid-chars))
  (next [this] (.next this batch-size))
  (next [this n]
    (when  (> (+ @char-pointer (* n segment-length)) max-char-pointer)
      (throw (NoSuchElementException.)))
    ;; Allocate space:
    (let [input (zeros [n (count valid-chars) segment-length])
          labels (zeros [n (count valid-chars) segment-length])]
      (dotimes [i n]
        (let [start-idx @char-pointer
              end-idx (swap! char-pointer + segment-length)]
          (put-scalar input [i (char-to-idx (aget input-chars start-idx)) 0] 1.0)
          (doseq [c (range 1 segment-length)]
            (let [char-idx (char-to-idx (aget input-chars (+ start-idx c)))]
              (put-scalar input [i char-idx c] 1.0)
              (put-scalar labels [i char-idx (dec c)] 1.0)))
          (put-scalar labels [i  (char-to-idx (aget input-chars end-idx)) (dec segment-length)] 1.0)))
      (data-set input labels)))
  (hasNext [this]
    (<= @char-pointer (- max-char-pointer (* batch-size segment-length))))
  (numExamples [this] n-segments)
  (reset [this] (reset! char-pointer 0))
  (totalExamples [this] n-segments) 
  (totalOutcomes [this] (count valid-chars)))

(defmethod print-method LRCharDataSetIterator [iter ^java.io.Writer w]
  (.write w (str "#LRCharDataSetIterator["
                 "n-segments=" (:n-segments iter) 
                 ",batch-size=" (:batch-size iter)
                 ",char-pointer=" @(:char-pointer iter)
                 "]")))

(defn lr-character-iterator 
  "Reifies a Datasetiterator iterating over text segments in a string from left to right."
  ([string]
   (lr-character-iterator string {}))
  ([^String string {:keys [batch-size ;; number of text segments per batch
                          segment-length ;; number of characters per text segment
                          n-segments ;; number of segments to iterate. Leave unspecified to return the maximum number of segments
                          valid-chars] ;; set of allowed characters. Leave unspecified to allow all characters.
                   :or {batch-size 100
                        segment-length 100}
                   :as opts}]
  (let [valid-chars (into #{} (or valid-chars string))
        chars (char-array (if valid-chars (filter valid-chars string) string))
        max-char-pointer (dec (count chars))
        max-segments (Math/floorDiv (long max-char-pointer) (long segment-length))]
    (when (and n-segments (> n-segments max-segments)) 
      (throw (IllegalArgumentException. (str "n-segments exceeds number of available segments " max-segments))))
    (when (and n-segments (not (zero? (mod n-segments batch-size))))
      (throw (IllegalArgumentException. (str "n-segments must be a multiple of batch-size"))))
    (println "data has" (count string) "characters," (count valid-chars) "unique.")
    (LRCharDataSetIterator. valid-chars 
                            (index-map valid-chars) 
                            chars 
                            segment-length 
                            (or n-segments max-segments)
                            batch-size 
                            (atom 0)
                            max-char-pointer))))

(defn- char-indices [example features-array]
  (for [pos (range (first (shape features-array)))]
    (for [fi (range (second (shape features-array)))]
      (first (get-scalar features-array [example fi pos])))))

(defn- binary-feature->char-idx [bf]
  (let [idx (remove #(zero? (first %)) (map vector bf (range)))]
    ;; (assert (= 1 (count idx)))
    (second (first idx))))

(defn- batch-examples [batch idx-to-char]
  (for [example (range (num-examples batch))]
    (let [features-array (get-features batch)
          labels-array (get-labels batch)]
      {:input (apply str (map #(idx-to-char (binary-feature->char-idx %)) 
                              (char-indices example features-array)))
       :output (apply str (map #(idx-to-char (binary-feature->char-idx %)) 
                               (char-indices example labels-array)))})))

(defn example-seq
  "Note: NOT thread safe!"
  [^LRCharDataSetIterator iter]
  (.reset iter)
  (let [idx-to-char (zipmap (map (:char-to-idx iter)
                                 (:valid-chars iter))
                            (:valid-chars iter))]
    (mapcat #(batch-examples % idx-to-char) 
            (iterator-seq iter))))



(comment

  (def lr-shakespeare-iterator (lr-character-iterator (shakespeare) {}))

  (:n-segments lr-shakespeare-iterator)
  ;; => 55898

  (first (example-seq lr-shakespeare-iterator))
  ;; => {:input "﻿The Project Gutenberg EBook of The Complete Works of William Shakespeare, by\r\nWilliam Shakespeare\r\n",
  ;;     :output "The Project Gutenberg EBook of The Complete Works of William Shakespeare, by\r\nWilliam Shakespeare\r\n\r"}
  
  )
