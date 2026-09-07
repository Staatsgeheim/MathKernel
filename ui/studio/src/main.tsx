import { createRoot } from 'react-dom/client';
import { App } from './app/App';
import { ViewBoundary } from './app/components';
import './app/styles.css';
createRoot(document.getElementById('root')!).render(
  <ViewBoundary>
    <App />
  </ViewBoundary>,
);
