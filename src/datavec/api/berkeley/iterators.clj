(ns dl4clj.datavec.api.berkeley.iterators
  (:import [org.datavec.api.berkeley
            Iterators
            Iterators$FilteredIterator ;; only returns items of a base iterator that pass a filter
            Iterators$IteratorIterator ;; wraps 2 level iteration scenario in an iterator
            Iterators$Transform ;; wraps base iterator with transform fn
            Iterators$TransformingIterator
            Filter]))
