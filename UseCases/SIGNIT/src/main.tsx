import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import Rules from './Rules'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import './index.css'

const router = createBrowserRouter([
  { path: '/', element: <App /> },
  { path: '/rules', element: <Rules /> },
]);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)
