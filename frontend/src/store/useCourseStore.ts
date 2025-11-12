import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { CourseItem } from '@/types'

interface CourseStore {
  courseItems: CourseItem[]
  addItem: (item: CourseItem) => void
  removeItem: (title: string) => void
  clearCourse: () => void
  reorderItems: (startIndex: number, endIndex: number) => void
}

export const useCourseStore = create<CourseStore>()(
  persist(
    (set) => ({
      courseItems: [],

      addItem: (item) =>
        set((state) => {
          // Check if item already exists
          if (state.courseItems.some((i) => i.title === item.title)) {
            return state
          }
          return { courseItems: [...state.courseItems, item] }
        }),

      removeItem: (title) =>
        set((state) => ({
          courseItems: state.courseItems.filter((item) => item.title !== title),
        })),

      clearCourse: () => set({ courseItems: [] }),

      reorderItems: (startIndex, endIndex) =>
        set((state) => {
          const result = Array.from(state.courseItems)
          const [removed] = result.splice(startIndex, 1)
          result.splice(endIndex, 0, removed)
          return { courseItems: result }
        }),
    }),
    {
      name: 'festmoment-course-storage',
    }
  )
)
